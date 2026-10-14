# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise weight ownership across submodule execution and skipped blocks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.block_discovery import get_blocks_from_dit
from vllm_omni.diffusion.offloader.config import OffloadStrategy, offload_streams_blocks, resolve_offload
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import current_omni_platform
from vllm_omni.diffusion.offloader.submodule_offload_backend import SubmoduleOffloadBackend

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class TinyBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.adaln_proj = nn.Linear(4, 4)
        self.attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 4))
        self.norm = nn.LayerNorm(4)
        with torch.no_grad():
            for param in self.parameters():
                param.add_(index * 0.05)

    def forward(self, x):
        return x + self.mlp(self.attn(self.adaln_proj(self.norm(x))))


class TinyDiT(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, count):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(index) for index in range(count)])

    def forward(self, x, indices=None):
        for index in range(len(self.blocks)) if indices is None else indices:
            x = self.blocks[index](x)
        return x


class ReferenceBlock(TinyBlock):
    def __init__(self, index):
        super().__init__(index)
        self.layer_index = index
        self.calls = []

    def forward(self, hidden_states, *, reference_kv_tier1_state=None, reference_kv_compact=False):
        from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import MiniMaxH3ReferenceKVTier2State

        result = super().forward(hidden_states)
        state = reference_kv_tier1_state
        if state is None:
            return result
        self.calls.append(state.current_step)
        k = result.unsqueeze(1)
        if isinstance(state, MiniMaxH3ReferenceKVTier2State):
            state.process_post_parallel_layer(
                self.layer_index,
                k.unsqueeze(0),
                (k + 1).unsqueeze(0),
                compact=reference_kv_compact,
                parallel_strategy="none",
            )
        elif reference_kv_compact:
            state.take_cached_reference_layer(self.layer_index, k)
        else:
            state.process_layer(self.layer_index, k, k + 1)
        return result


class ReferenceDiT(TinyDiT):
    def __init__(self, count):
        super().__init__(count)
        self.blocks = nn.ModuleList([ReferenceBlock(index) for index in range(count)])

    def forward(self, hidden_states, **kwargs):
        for block in self.blocks:
            hidden_states = block(hidden_states, **kwargs)
        return hidden_states


@pytest.fixture
def runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)


@pytest.mark.parametrize("count", [3, 4])
def test_repeated_forward_skipped_blocks_and_restore(runtime, count):
    torch.manual_seed(17)
    pipeline = nn.Module()
    pipeline.transformer = TinyDiT(count)
    baseline = deepcopy(pipeline.transformer)
    backend = SubmoduleOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.SUBMODULE,
            pin_cpu_memory=False,
            dlo_use_allgather=False,
            components=frozenset({"dit"}),
        ),
        torch.device("cpu"),
    )
    backend.enable(pipeline)
    hooks = backend._all_hook_groups[0]
    assert len(hooks) == count * 3
    assert all(hook.gpu_buffers is hooks[0].gpu_buffers for hook in hooks)
    assert len(hooks[0].gpu_buffers) == 2
    largest = max(
        sum(t.numel() for t in unit.parameters())
        for block in baseline.blocks
        for unit in (block.adaln_proj, block.attn, block.mlp)
    )
    assert hooks[0].gpu_buffers[0][torch.float32].numel() == largest
    assert all(block.norm.weight.numel() == 4 for block in pipeline.transformer.blocks)

    x = torch.randn(2, 4)
    with torch.no_grad():
        # Cache-DiT may execute only the front/back layers of a physical stack.
        for indices in (None, [0, count - 1], None, [1, count - 1], None):
            torch.testing.assert_close(pipeline.transformer(x, indices), baseline(x, indices))
    backend.disable()
    for name, param in pipeline.transformer.named_parameters():
        torch.testing.assert_close(param, dict(baseline.named_parameters())[name])


def test_discover_physical_cachedit_layers():
    dit = TinyDiT(4)
    originals = list(dit.blocks)
    wrapper = nn.Module()
    wrapper.transformer_blocks = dit.blocks
    dit.blocks = nn.ModuleList([wrapper])
    attrs, blocks = get_blocks_from_dit(dit)
    assert attrs == ["blocks"]
    assert blocks == originals
    assert dit.blocks[0] is wrapper


def test_submodule_config_and_factory(runtime, monkeypatch):
    od_config = SimpleNamespace(enable_submodule_offload=True)
    resolved = resolve_offload(od_config)
    assert resolved.strategy is OffloadStrategy.SUBMODULE
    assert resolved.any_allgather
    assert offload_streams_blocks(od_config)
    assert OffloadConfig.from_od_config(od_config).strategy is OffloadStrategy.SUBMODULE
    monkeypatch.setattr(current_omni_platform, "supports_cpu_offload", lambda: True)
    monkeypatch.setattr(current_omni_platform, "get_device_count", lambda: 1)
    backend = get_offload_backend(od_config, torch.device("cpu"))
    assert isinstance(backend, SubmoduleOffloadBackend)


def test_incompatible_auxiliary_stack_keeps_block_hooks(runtime):
    backend = SubmoduleOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.SUBMODULE, pin_cpu_memory=False, dlo_use_allgather=False),
        torch.device("cpu"),
    )
    blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    hooks = backend._install_hook_group(blocks, "dit")
    assert len(hooks) == 2
    backend.disable()


@pytest.mark.parametrize("tier", [1, 2])
@pytest.mark.parametrize("compact", [False, True])
def test_real_cachedit_wrapper_with_reference_kv_and_submodule(runtime, tier, compact):
    _run_cachedit_reference_offload(torch.device("cpu"), tier, compact=compact)


def _run_cachedit_reference_offload(device, tier, *, compact=True):
    from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern
    from cache_dit.caching.block_adapters import FakeDiffusionPipeline

    from vllm_omni.diffusion.models.minimax_h3.reference_kv_cachedit import MiniMaxH3CachedAdapter
    from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
        MiniMaxH3ReferenceKVTier1State,
        MiniMaxH3ReferenceKVTier2State,
    )

    torch.manual_seed(19)
    model = ReferenceDiT(4)
    originals = list(model.blocks)
    expected_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    baseline = deepcopy(model).to(device)
    MiniMaxH3CachedAdapter.apply(
        BlockAdapter(
            pipe=type("ReferencePipeline", (FakeDiffusionPipeline,), {"_is_cached": False})(model),
            transformer=model,
            blocks=model.blocks,
            blocks_name="blocks",
            forward_pattern=ForwardPattern.Pattern_3,
            check_forward_pattern=False,
            has_separate_cfg=False,
        ),
        cache_config=DBCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=1,
            max_warmup_steps=0,
            num_inference_steps=8,
            residual_diff_threshold=100.0,
            max_continuous_cached_steps=-1,
        ),
    )
    pipeline = nn.Module()
    pipeline.transformer = model
    backend = SubmoduleOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.SUBMODULE,
            pin_cpu_memory=device.type != "cpu",
            dlo_use_allgather=False,
            components=frozenset({"dit"}),
        ),
        device,
    )
    backend.enable(pipeline)
    assert len(backend._all_hook_groups[0]) == 12
    kwargs = dict(
        num_layers=4,
        global_reference_rows=2,
        device=device,
        refresh_interval=4,
        pin_memory=device.type != "cpu",
        skip_reference_projection=compact,
    )
    if tier == 2:
        state = MiniMaxH3ReferenceKVTier2State(
            **kwargs,
            ring_size=2,
            host_quantization="int8" if device.type == "npu" else "none",
        )
    else:
        state = MiniMaxH3ReferenceKVTier1State(**kwargs)
    mask = torch.tensor([True, False, True, False], device=device)
    state.set_global_reference_mask(mask)
    x = torch.randn(4, 4, device=device)
    with torch.no_grad():
        expected_output = baseline(x)
    try:
        with torch.no_grad():
            for step in range(8):
                state.begin_step(step)
                state.set_local_reference_mask(mask)
                reuse_compact = state.should_skip_reference_projection
                result = model(
                    x[~mask] if reuse_compact else x,
                    reference_kv_tier1_state=state,
                    reference_kv_compact=reuse_compact,
                )
                assert torch.isfinite(result).all()
                torch.testing.assert_close(
                    result,
                    expected_output[~mask] if reuse_compact else expected_output,
                    atol=1e-4,
                    rtol=1e-4,
                )
                if device.type == "npu":
                    torch.npu.synchronize()
                state.end_step()
        assert originals[0].calls == list(range(8))
        assert originals[-1].calls == list(range(8))
        assert originals[1].calls == ([0, 1, 4, 5] if compact else [0, 4])
        assert originals[2].calls == ([0, 1, 4, 5] if compact else [0, 4])
        assert state.stats.captured_layers == 8
        assert state.stats.block_cached_layers == (8 if compact else 12)
        if device.type == "npu" and tier == 2:
            assert state.effective_host_quantization == "int8"
            assert state.host_pool.is_pinned()
            assert state.stats.dequantized_layers > 0
    finally:
        state.close()
        backend.disable()
    # The wrapper changes state-dict prefixes; compare the physical blocks.
    for index, block in enumerate(originals):
        for name, param in block.named_parameters():
            torch.testing.assert_close(param.cpu(), expected_parameters[f"blocks.{index}.{name}"])


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
def test_npu_submodule_cachedit_int8_reference_ring():
    _run_cachedit_reference_offload(torch.device("npu:0"), 2)

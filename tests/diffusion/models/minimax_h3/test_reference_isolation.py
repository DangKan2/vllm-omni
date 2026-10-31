# SPDX-License-Identifier: Apache-2.0
"""CPU-only isolation contracts; no model weights, vLLM init or NPU context."""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "vllm_omni/diffusion/models/minimax_h3"
spec = importlib.util.spec_from_file_location("_h3_isolation_cpu_test", MODEL / "reference_isolation.py")
isolation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation_module)
Isolation = isolation_module.MiniMaxH3ReferenceIsolation
ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_ISOLATION"


@dataclass
class Metadata:
    attn_mask: object = None
    packed_padding: object = None
    video_layout: object = None
    extra: dict = field(default_factory=dict)
    full_attn_spans: object = None


def make_state(refs=(1, 4), used=7, total=9):
    return Isolation(
        torch.tensor(refs, dtype=torch.long),
        sequence_rows=total,
        used_rows=used,
        condition_timestep=0.999,
        device=torch.device("cpu"),
    )


def dense(q, k, v, metadata):
    assert metadata.attn_mask is None
    assert metadata.extra["max_seqlen_q"] == q.shape[1]
    assert metadata.extra["max_seqlen_k"] == k.shape[1]
    assert metadata.extra["cu_seqlens_q"].tolist() == [0, q.shape[1], q.shape[1]]
    assert metadata.extra["cu_seqlens_k"].tolist() == [0, k.shape[1], k.shape[1]]
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


@pytest.mark.parametrize("value", [None, False, True])
def test_config_uses_serialized_option_and_ignores_environment(value, monkeypatch):
    monkeypatch.setenv(ENV, "1" if value is not True else "0")
    config = SimpleNamespace(kv_isolate=value)
    assert isolation_module.reference_isolation_enabled(config) is (value is True)
    assert isolation_module.reference_isolation_enabled(SimpleNamespace()) is False


@pytest.mark.parametrize("value", ["1", "true", "reference_only", 0, 1, [], {}])
def test_invalid_config(value):
    with pytest.raises(ValueError, match="boolean"):
        isolation_module.reference_isolation_enabled(SimpleNamespace(kv_isolate=value))


@pytest.mark.parametrize("refs", [(), (0,), (1, 4), (0, 2, 6), tuple(range(7))])
def test_four_regions_match_explicit_oracle_and_padding(refs):
    torch.manual_seed(71)
    state = make_state(refs)
    q, k, v = [torch.randn(1, 9, 2, 4, dtype=torch.float64) for _ in range(3)]
    metadata = Metadata(extra={"laser_input_scale": 4.0})
    out = state.attend(q, k, v, metadata, compact=False, dense_forward=dense)
    allowed = torch.ones(7, 7, dtype=torch.bool)
    reference = torch.zeros(7, dtype=torch.bool)
    reference[list(refs)] = True
    allowed[reference] = reference
    expected = F.scaled_dot_product_attention(
        q[:, :7].transpose(1, 2),
        k[:, :7].transpose(1, 2),
        v[:, :7].transpose(1, 2),
        attn_mask=allowed,
    ).transpose(1, 2)
    torch.testing.assert_close(out[:, :7], expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(out[:, 7:]) == 0
    assert metadata.extra == {"laser_input_scale": 4.0}
    assert state.full_layer_calls == 1


def test_reference_cannot_read_any_dynamic_text_audio_or_latent():
    torch.manual_seed(9)
    state = make_state()
    q, k, v = [torch.randn(1, 9, 2, 4) for _ in range(3)]
    original = state.attend(q, k, v, Metadata(), compact=False, dense_forward=dense)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, state.dynamic_positions] += 12
    changed_v[:, state.dynamic_positions] -= 21
    # Padding also must never influence valid outputs.
    changed_k[:, 7:] = 10000
    changed_v[:, 7:] = -10000
    changed = state.attend(q, changed_k, changed_v, Metadata(), compact=False, dense_forward=dense)
    torch.testing.assert_close(
        original[:, state.reference_positions], changed[:, state.reference_positions], rtol=0, atol=0
    )
    assert not torch.allclose(original[:, state.dynamic_positions], changed[:, state.dynamic_positions])


def test_compact_cached_reference_prefix_matches_full_isolated_target_outputs():
    torch.manual_seed(31)
    state = make_state()
    q, k, v = [torch.randn(1, 9, 2, 4, dtype=torch.float64) for _ in range(3)]
    full = state.attend(q, k, v, Metadata(), compact=False, dense_forward=dense)
    order = torch.cat((state.reference_positions, state.dynamic_positions))
    compact_q = torch.cat((q[:, state.dynamic_positions], torch.zeros(1, 2, 2, 4)), dim=1)
    metadata = Metadata(extra={"minimax_h3_compact_reference_rows": 2})
    compact = state.attend(compact_q, k[:, order], v[:, order], metadata, compact=True, dense_forward=dense)
    torch.testing.assert_close(compact[:, :5], full[:, state.dynamic_positions], rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(compact[:, 5:]) == 0
    assert state.compact_layer_calls == 1


def test_static_reference_kv_and_cached_reuse_across_multiple_layers_and_steps():
    torch.manual_seed(13)
    state = make_state()
    reference = state.reference_positions
    dynamic = state.dynamic_positions
    anchor = torch.randn(1, 9, 8, dtype=torch.float64)
    weights = [torch.randn(8, 24, dtype=torch.float64) * 0.15 for _ in range(4)]
    cached = []
    for step in range(4):
        hidden = anchor.clone()
        # Model text/audio/latent and their timestep modulation all change.
        hidden[:, dynamic] += step * 0.19
        compact_hidden = hidden[:, dynamic].clone()
        for layer, weight in enumerate(weights):
            q, k, v = [(hidden @ weight).view(1, 9, 3, 2, 4)[:, :, i] for i in range(3)]
            if step == 0:
                cached.append((k[:, reference].clone(), v[:, reference].clone()))
            else:
                torch.testing.assert_close(k[:, reference], cached[layer][0], rtol=0, atol=0)
                torch.testing.assert_close(v[:, reference], cached[layer][1], rtol=0, atol=0)
                cq, ck, cv = [(compact_hidden @ weight).view(1, 5, 3, 2, 4)[:, :, i] for i in range(3)]
                out = state.attend(
                    cq,
                    torch.cat((cached[layer][0], ck), 1),
                    torch.cat((cached[layer][1], cv), 1),
                    Metadata(extra={"minimax_h3_compact_reference_rows": 2}),
                    compact=True,
                    dense_forward=dense,
                )
                compact_hidden += out.reshape(1, 5, 8)
            full = state.attend(q, k, v, Metadata(), compact=False, dense_forward=dense)
            hidden += full.reshape(1, 9, 8)
            if step:
                torch.testing.assert_close(compact_hidden, hidden[:, dynamic], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("refs", [(7,), (-1,), (2, 1), (1, 1)])
def test_invalid_reference_ranges(refs):
    with pytest.raises(ValueError):
        make_state(refs)


@pytest.mark.parametrize("timestep", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_reference_timestep(timestep):
    with pytest.raises(ValueError):
        Isolation(
            torch.tensor([1]), sequence_rows=8, used_rows=7, condition_timestep=timestep, device=torch.device("cpu")
        )


def test_layout_and_explicit_mask_fail_closed():
    state = make_state()
    with pytest.raises(ValueError, match="one request"):
        state.validate_presentation(sequence_rows=9, used_rows=7, num_requests=2)
    with pytest.raises(ValueError, match="layout changed"):
        state.validate_presentation(sequence_rows=10, used_rows=7, num_requests=1)
    q = torch.randn(1, 9, 2, 4)
    with pytest.raises(ValueError, match="explicit attention mask"):
        state.attend(q, q, q, Metadata(attn_mask=torch.ones(1, 9)), compact=False, dense_forward=dense)
    with pytest.raises(ValueError, match="reference-prefix"):
        state.attend(q, q, q, Metadata(), compact=True, dense_forward=dense)


def test_request_local_indices_metadata_and_close():
    one, two = make_state((1, 4)), make_state((2, 3))
    q = torch.randn(1, 9, 2, 4)
    one.attend(q, q, q, Metadata(), compact=False, dense_forward=dense)
    assert len(one._packed_lengths) == 2
    assert not two._packed_lengths
    one.close()
    assert not one._packed_lengths and one.reference_positions.numel() == 0
    with pytest.raises(RuntimeError, match="closed"):
        one.attend(q, q, q, Metadata(), compact=False, dense_forward=dense)
    two.attend(q, q, q, Metadata(), compact=False, dense_forward=dense)


def load_function(path, name, extra=None, class_name=None):
    tree = ast.parse(path.read_text())
    if class_name is not None:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    namespace = {
        "torch": torch,
        "is_forward_context_available": lambda: False,
        "logger": SimpleNamespace(info_once=lambda *a: None),
        **(extra or {}),
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(fn), str(path), "exec"), namespace)
    return namespace[name]


def test_model_public_keyword_guard_accepts_request_isolation():
    path = MODEL / "minimax_h3_transformer.py"
    tree = ast.parse(path.read_text())
    contract = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_FORWARD_SUPPORTED_KWARGS" for t in node.targets)
    )
    supported = eval(compile(ast.Expression(contract), str(path), "eval"))
    guard = load_function(
        path,
        "forward",
        {
            "_FORWARD_SUPPORTED_KWARGS": supported,
            "_required_kwarg": load_function(path, "_required_kwarg"),
        },
        class_name="MiniMaxH3DiTModel",
    )
    # The new request kwarg must pass the real public guard; validation then
    # reaches the deliberately omitted required x, before any tensor work.
    with pytest.raises(ValueError, match="requires kwarg 'x'"):
        guard(SimpleNamespace(), reference_isolation=make_state())


def layer_fixture(name="FLASH_ATTN"):
    calls = []
    backend = SimpleNamespace(forward=lambda q, k, v, m: (calls.append("dense"), dense(q, k, v, m))[1])
    sparse = SimpleNamespace(dense_fallback=backend, forward=lambda *a: pytest.fail("sparse path must not run"))
    layer = SimpleNamespace(
        _scheduler_paged_kv=False,
        paged_kv_cache_role=None,
        use_ring=False,
        causal=False,
        qkv_layout="BSND",
        attn_backend=SimpleNamespace(get_name=lambda: name),
        attention=sparse if name == "RAINFUSION_ATTN" else backend,
        _active_paged_kv_adapter=lambda: None,
        _with_kv_cache_dtype=lambda m: m,
        _run_local_attention=backend.forward,
    )
    strategy = SimpleNamespace(
        name="ulysses",
        pre_attention=lambda q, k, v, m: (calls.append("pre"), (q, k, v, m, None))[1],
        post_attention=lambda out, ctx: (calls.append("post"), out)[1],
    )
    layer._get_active_parallel_strategy = lambda: strategy
    layer._no_parallel_strategy = strategy
    path = ROOT / "vllm_omni/diffusion/attention/layer.py"
    impl = load_function(path, "_reference_isolation_dense_impl")
    layer._reference_isolation_dense_impl = lambda: impl(layer)
    return layer, strategy, calls, load_function(path, "_forward_impl")


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "RAINFUSION_ATTN", "SDPA"])
def test_actual_layer_hook_runs_after_ulysses_and_before_reverse_exchange(backend):
    state = make_state()
    q = torch.randn(1, 9, 2, 4)
    metadata = Metadata(extra={"minimax_h3_reference_isolation": (state, False)})
    layer, strategy, calls, forward = layer_fixture(backend)
    out = forward(layer, q, q, q, metadata)
    assert calls == ["pre", "dense", "dense", "post"]
    expected = state.attend(q, q, q, Metadata(), compact=False, dense_forward=dense)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("unsupported", ["ring", "allgather_kv", "causal", "backend"])
def test_layer_rejects_unsupported_path_before_collectives(unsupported):
    layer, strategy, calls, forward = layer_fixture()
    if unsupported == "ring":
        layer.use_ring = True
    elif unsupported == "causal":
        layer.causal = True
    elif unsupported == "backend":
        layer.attn_backend.get_name = lambda: "FASTVIDEO_VSA"
    else:
        strategy.name = unsupported
    state = make_state()
    q = torch.randn(1, 9, 2, 4)
    with pytest.raises(ValueError):
        forward(layer, q, q, q, Metadata(extra={"minimax_h3_reference_isolation": (state, False)}))
    assert not calls


def test_disabled_layer_preserves_original_dispatch_and_metadata_identity():
    layer, strategy, calls, forward = layer_fixture()
    q = torch.randn(1, 9, 2, 4)
    metadata = Metadata()
    seen = []
    expected = torch.randn_like(q)
    layer._run_local_attention = lambda q, k, v, m: (seen.append(m), expected)[1]
    result = forward(layer, q, q, q, metadata)
    assert result is expected and seen == [metadata]
    assert calls == ["pre", "post"]


def test_model_packed_attention_carries_isolation_and_padding_to_actual_layer_hook():
    state = make_state()
    q = torch.randn(9, 2, 4)
    layer, strategy, calls, forward = layer_fixture("RAINFUSION_ATTN")
    layer.attn_backend.supports_packed_mask_free = lambda: False
    layer.attn_backend.supports_prefix_kv_slicing = False
    packed = load_function(
        MODEL / "minimax_h3_transformer.py",
        "_run_packed_attention",
        {
            "AttentionMetadata": Metadata,
            "PackedPaddingMetadata": lambda **kw: SimpleNamespace(**kw),
            "MINIMAX_H3_LASER_INPUT_SCALE": 4.0,
            "_ring_sequence_parallel_is_active": lambda attention: False,
        },
    )

    # Keep the same layer instance for capability checks and actual invocation.
    class BoundLayer:
        def __getattr__(self, name):
            return getattr(layer, name)

        def __call__(self, q, k, v, metadata):
            return forward(layer, q, k, v, metadata)

    owner = SimpleNamespace(attention=BoundLayer())
    out = packed(
        owner,
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 7, 9], dtype=torch.int32),
        max_seqlen=7,
        packed_total=9,
        reference_isolation=state,
    )
    expected = state.attend(q[None], q[None], q[None], Metadata(), compact=False, dense_forward=dense)
    torch.testing.assert_close(out, expected[0], rtol=0, atol=0)
    assert calls == ["pre", "dense", "dense", "post"]


def load_cpu_cache_module():
    # Load the actual existing manager with only logging/platform integration
    # substituted, avoiding vLLM import-time NPU discovery. Its cache/ring,
    # capture, restore and accounting implementations remain unchanged.
    path = MODEL / "reference_kv_tier1.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and ((node.module or "").startswith(("vllm.", "vllm_omni.")) or node.module == "reference_kv_diagnostics")
        )
    ]
    module = types.ModuleType("_h3_existing_cpu_cache")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        {
            "init_logger": logging.getLogger,
            "current_omni_platform": SimpleNamespace(),
            "timing_span": lambda *a, **kw: nullcontext(),
        }
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def test_actual_tier2_manager_populates_once_and_matches_recomputed_isolated_layers():
    module = load_cpu_cache_module()
    torch.manual_seed(19)
    isolation = make_state()
    mask = torch.zeros(9, dtype=torch.bool)
    mask[isolation.reference_positions] = True
    cache = module.MiniMaxH3ReferenceKVTier2State(
        num_layers=3,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=0,
        ring_size=2,
        pin_memory=False,
        skip_reference_projection=True,
    )
    full_anchor = torch.randn(1, 9, 8, dtype=torch.float64)
    weights = [torch.randn(8, 24, dtype=torch.float64) * 0.1 for _ in range(3)]
    for step in range(3):
        cache.begin_step(step)
        cache.set_global_reference_mask(mask)
        cache.set_local_reference_mask(mask)
        full = full_anchor.clone()
        full[:, isolation.dynamic_positions] += 0.25 * step
        hidden = full.clone() if step == 0 else full[:, cache.global_non_reference_positions].clone()
        for layer, weight in enumerate(weights):
            q, k, v = [(hidden @ weight).view(1, hidden.shape[1], 3, 2, 4)[:, :, i] for i in range(3)]
            k, v = cache.process_post_parallel_layer(layer, k, v, compact=step > 0, parallel_strategy="ulysses")
            output = isolation.attend(
                q,
                k,
                v,
                Metadata(extra={"minimax_h3_compact_reference_rows": 2} if step else {}),
                compact=step > 0,
                dense_forward=dense,
            )
            hidden += output.reshape_as(hidden)
            fq, fk, fv = [(full @ weight).view(1, 9, 3, 2, 4)[:, :, i] for i in range(3)]
            full += isolation.attend(fq, fk, fv, Metadata(), compact=False, dense_forward=dense).reshape_as(full)
            expected = full if step == 0 else full[:, cache.global_non_reference_positions]
            torch.testing.assert_close(hidden, expected, rtol=1e-11, atol=1e-11)
        cache.end_step()
    assert cache.stats.refresh_steps == 1
    assert cache.stats.captured_layers == 3 and cache.stats.substituted_layers == 6
    cache.close()


@pytest.mark.parametrize("failure", ["forward", "cache_create", "cache_close"])
def test_denoise_failures_close_isolation_and_remove_request_kwargs(failure):
    captured, received = [], []

    def factory(*args, **kwargs):
        state = Isolation(*args, **kwargs)
        captured.append(state)
        return state

    branch = SimpleNamespace(
        img_pos=torch.tensor([1, 2]),
        update_mask=torch.tensor([False, True]),
        seq_len=4,
        used_len=4,
        update_mask_dev=torch.tensor([False, True]),
        audio_update_mask_dev=torch.tensor([True]),
        static_kwargs={},
        num_visual_reference_rows=1,
    )

    def forward_kwargs(**kwargs):
        received.append(kwargs)
        return {**branch.static_kwargs, **kwargs}

    branch.forward_kwargs = forward_kwargs

    def close():
        if failure == "cache_close":
            raise RuntimeError("injected cache_close failure")

    cache = SimpleNamespace(begin_step=lambda *a, **k: None, close=close)

    class Model:
        od_config = SimpleNamespace(kv_isolate=True)

        def create_reference_kv_tier1_state(self, **kwargs):
            if failure == "cache_create":
                raise RuntimeError("injected cache_create failure")
            return cache

        def __call__(self, **kwargs):
            raise RuntimeError("injected forward failure")

    loop = load_function(
        MODEL / "denoise_loop.py",
        "minimax_h3_denoise_loop",
        {
            "current_omni_platform": SimpleNamespace(is_npu=lambda: False),
            "reference_isolation_enabled": isolation_module.reference_isolation_enabled,
            "MiniMaxH3ReferenceIsolation": factory,
            "minimax_h3_prepare_denoise_rows": lambda **k: (torch.zeros(2, 3), torch.zeros(1, 3), None, None),
            "minimax_h3_prepare_edit_rows": lambda rows, *args, **kw: (rows, None),
            "create_h3_sample_solver": lambda *a: None,
            "check_request_cancellation": lambda **k: None,
            "minimax_h3_publish_denoise_progress": lambda *a: None,
            "nullcontext": nullcontext,
            "ReferenceKVRefreshRequired": type("Refresh", (Exception,), {}),
            "MINIMAX_H3_IMGVID_COND_TIMESTEP": 0.999,
            "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP": 1.0,
        },
    )
    with pytest.raises(RuntimeError, match=failure):
        loop(
            model=Model(),
            positive=branch,
            initial_video_rows=torch.zeros(2, 3),
            initial_audio_rows=torch.zeros(1, 3),
            keyframe_cond_rows=None,
            sigmas_video=[0.1, 0.0],
            sigmas_audio=[0.1, 0.0],
            device=torch.device("cpu"),
            imgvid_cond_noise_aug_for_inference=0.2,
        )
    if received:
        assert received[0]["imgvid_cond_timestep"] == 0.2
    assert captured[0].closed and not branch.static_kwargs

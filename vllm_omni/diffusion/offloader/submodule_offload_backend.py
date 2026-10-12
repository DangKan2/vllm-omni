# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sub-module level weight offload backend for DiT blocks.

Extends the DLO double-buffer + AllGather + mmap infrastructure to the
sub-module level (attn / mlp / adaln_proj) within each DiT block.  While
one sub-module computes, the next sub-module's weights are prefetched,
reducing GPU buffer size from block-level (~1.2 GB) to sub-module level
(~0.5 GB) and overlapping H2D with finer-grained compute.

Pipeline per block:
    AdaLN_N  ->  prefetch Attention_{N+1}
    Attention_N  ->  prefetch MLP_N (current block)
    MLP_N  ->  prefetch AdaLN_{N+1}

The circular ring spans all 50 blocks × 3 sub-modules = 150 hooks.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.vllm_platforms import current_omni_platform

from vllm_omni.diffusion.offloader.component_utils import move_non_block_state_to_device
from vllm_omni.diffusion.offloader.config import DIT_COMPONENT
from vllm_omni.diffusion.offloader.tensor_utils import (
    TensorStorageSpec,
    clear_block_storage,
    describe_tensor_storage,
    flatten_physical_storage,
    group_named_tensors_by_dtype,
    is_materialized_tensor,
    materialization_probe,
    module_materialization_probe,
    set_tensor_storage,
    tensor_storage_metadata,
)
from vllm_omni.platforms import current_omni_platform as _current_omni_platform

from .distributed_layerwise_backend import DistributedLayerwiseOffloadHook

logger = init_logger(__name__)

HOOK_NAME = "submodule_offload"


class SubmoduleOffloadHook:
    """Per-sub-module hook for sub-module level weight streaming.

    Registered on one sub-module (attn / mlp / adaln_proj) of a DiT block.
    In pre_forward: wait for own weights, then prefetch the NEXT sub-module
    in the pipeline ring.  In post_forward: release own GPU weights.

    Attributes:
        sub_module: The nn.Module this hook is registered on (e.g. block.attn).
        next_hook: The next SubmoduleOffloadHook in the circular ring.
        prev_hook: The previous SubmoduleOffloadHook (for wait fallback).
    """

    def __init__(
        self,
        sub_module: nn.Module,
        sub_name: str,
        device: torch.device,
        dp_group: torch.distributed.ProcessGroup | None,
        dp_size: int,
        rank: int,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
        pin_memory: bool = True,
        shared_gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
        shared_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
    ) -> None:
        self.sub_module = sub_module
        self.sub_name = sub_name
        self.device = device
        self.dp_group = dp_group
        self.dp_size = dp_size
        self.rank = rank
        self.pin_memory = pin_memory

        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        if shared_gpu_buffers is not None:
            self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_gpu_buffers
            self._owns_buffers = False
        else:
            self.gpu_buffers = [None, None]
            self._owns_buffers = True
        self.ready_events: list[Any | None] = [None, None]

        if shared_shard_buffers is not None:
            self.gpu_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_shard_buffers
        else:
            self.gpu_shard_buffers = [None, None]

        self.cpu_shards: dict[torch.dtype, torch.Tensor] = {}
        self.metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
        self._ag_output_sizes: dict[torch.dtype, int] = {}

        self.current_slot = 0
        self._prefetched_slot: int | None = None

        self._prev_hook: SubmoduleOffloadHook | None = None
        self.next_hook: SubmoduleOffloadHook | None = None

        self._materialization_probe: torch.Tensor | None = None

        self._prefetch_done: Any | None = None
        self._hook_handles: list[Any] = []

        self.next_sub_parameters: dict[str, nn.Parameter] = {}
        self.next_sub_buffers: dict[str, torch.Tensor] = {}

        self._is_first_in_ring = False

    @property
    def is_materialized(self) -> bool:
        return self._materialization_probe is None or is_materialized_tensor(self._materialization_probe)

    # ------------------------------------------------------------------ #
    #  Weight sharding (reuses DLO _shard_and_pin logic)                 #
    # ------------------------------------------------------------------ #

    def _shard_and_pin(
        self,
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
    ) -> None:
        for dtype, named_weights in group_named_tensors_by_dtype(params, bufs).items():
            specs = describe_tensor_storage(named_weights, None)
            total_numel = sum(spec.storage_numel for spec in specs)

            if self.dp_size > 1:
                shard_size = (total_numel + self.dp_size - 1) // self.dp_size
                shard_start = self.rank * shard_size
                shard_end = min(shard_start + shard_size, total_numel)
                shard = torch.zeros(shard_size, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
            else:
                shard_size = total_numel
                shard = torch.zeros(shard_size, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
                shard_start = 0
                shard_end = total_numel

            current_offset = 0
            for spec in specs:
                self.metadata.setdefault(dtype, []).append(
                    tensor_storage_metadata(spec, current_offset)
                )
                overlap_start = max(current_offset, shard_start)
                overlap_end = min(current_offset + spec.storage_numel, shard_end)
                if overlap_start < overlap_end:
                    flat = flatten_physical_storage(spec.value, spec.storage_numel)
                    src_s = overlap_start - current_offset
                    src_e = overlap_end - current_offset
                    dst_s = overlap_start - shard_start
                    dst_e = overlap_end - shard_start
                    shard[dst_s:dst_e].copy_(flat[src_s:src_e])
                current_offset += spec.storage_numel

            self.cpu_shards[dtype] = shard
            if self.dp_size > 1:
                self._ag_output_sizes[dtype] = shard_size * self.dp_size
            else:
                self._ag_output_sizes[dtype] = total_numel

        clear_tensor_storage(chain(self.next_sub_parameters.values(), self.next_sub_buffers.values()))

    def _allocate_device_buffers(self) -> None:
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype in self.metadata:
                gpu_weights[dtype] = torch.empty(
                    self._ag_output_sizes[dtype], dtype=dtype, device=self.device
                )
            self.gpu_buffers[slot] = gpu_weights

        if self.dp_size > 1:
            for slot in range(2):
                shard_bufs: dict[torch.dtype, torch.Tensor] = {}
                for dtype in self.cpu_shards:
                    shard_bufs[dtype] = torch.empty(
                        self.cpu_shards[dtype].numel(), dtype=dtype, device=self.device
                    )
                self.gpu_shard_buffers[slot] = shard_bufs

    # ------------------------------------------------------------------ #
    #  Prefetch: H2D + AllGather (overlapped on dedicated streams)      #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def prefetch_layer(self, slot: int, non_blocking: bool = True) -> None:
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        evt = current_omni_platform.Event()
        gpu_weights = self.gpu_buffers[slot]
        assert gpu_weights is not None

        if self.dp_size <= 1 or self.dp_group is None:
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    gw = gpu_weights[dtype]
                    async_copy = non_blocking and cpu_shard.is_pinned()
                    gw[: cpu_shard.numel()].copy_(cpu_shard, non_blocking=async_copy)
                evt.record(self.copy_stream)
        else:
            shard_bufs = self.gpu_shard_buffers[slot]
            assert shard_bufs is not None
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    gpu_shard = shard_bufs[dtype][: cpu_shard.numel()]
                    gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)

            self.comm_stream.wait_stream(self.copy_stream)
            with current_omni_platform.stream(self.comm_stream):
                for dtype in self.cpu_shards:
                    gw = gpu_weights[dtype][: self._ag_output_sizes[dtype]]
                    shard_bufs_dtype = shard_bufs[dtype]
                    torch.distributed.all_gather_into_tensor(gw, shard_bufs_dtype, group=self.dp_group)

            evt.record(self.comm_stream)

        self.ready_events[slot] = evt
        self._prefetched_slot = slot
        self._prefetch_done = evt

        self._repoint_parameters(slot)

    def _repoint_parameters(self, slot: int) -> None:
        gpu_weights = self.gpu_buffers[slot]
        assert gpu_weights is not None
        for dtype, metas in self.metadata.items():
            buf = gpu_weights[dtype]
            for meta in metas:
                name = meta["name"]
                target = (
                    self.next_sub_parameters[name]
                    if name in self.next_sub_parameters
                    else self.next_sub_buffers[name]
                )
                physical = buf[meta["offset"]: meta["offset"] + meta["numel"]]
                view = torch.as_strided(physical, size=meta["shape"], stride=meta["stride"])
                set_tensor_storage(target, view)

    def _wait_for_weights(self, slot: int) -> None:
        evt = self.ready_events[slot]
        if evt is None and self._prev_hook is not None:
            evt = self._prev_hook.ready_events[slot]
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)

    # ------------------------------------------------------------------ #
    #  Offload: free device memory                                        #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def offload_layer(self) -> None:
        clear_block_storage(self.next_sub_parameters, self.next_sub_buffers, self._prefetch_done)
        self._prefetch_done = None

    # ------------------------------------------------------------------ #
    #  ModelHook interface                                               #
    # ------------------------------------------------------------------ #

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        if self._prev_hook is not None and self._prev_hook._prefetched_slot is not None:
            self.current_slot = self._prev_hook._prefetched_slot

        if not self.is_materialized and self._prev_hook is not None:
            self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
            self._prev_hook._wait_for_weights(self.current_slot)
        elif self._is_first_in_ring:
            if self._prev_hook is not None:
                self._prev_hook._wait_for_weights(self.current_slot)

        next_slot = 1 - self.current_slot
        if self.next_hook is not None:
            self.next_hook.prefetch_layer(next_slot, non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()
        return output

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def attach(self) -> None:
        self._hook_handles.append(
            self.sub_module.register_forward_pre_hook(self.pre_forward, prepend=True)
        )
        self._hook_handles.append(
            self.sub_module.register_forward_hook(self.post_forward, append=True)
        )

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def restore_to_cpu(self) -> None:
        if self.dp_size <= 1:
            for dtype, metas in self.metadata.items():
                cpu_shard = self.cpu_shards[dtype]
                for meta in metas:
                    name = meta["name"]
                    target = (
                        self.next_sub_parameters[name]
                        if name in self.next_sub_parameters
                        else self.next_sub_buffers[name]
                    )
                    restored = torch.empty_strided(
                        meta["shape"], meta["stride"], dtype=dtype, device="cpu"
                    )
                    src_start = meta["offset"]
                    src_end = src_start + meta["numel"]
                    restored_flattened = restored.flatten()
                    restored_flattened.copy_(cpu_shard[src_start:src_end])
                    set_tensor_storage(target, restored)
        else:
            from vllm_omni.diffusion.offloader.tensor_utils import restore_tensor_storage

            for dtype, metas in self.metadata.items():
                cpu_shard = self.cpu_shards[dtype]
                for meta in metas:
                    name = meta["name"]
                    target = (
                        self.next_sub_parameters[name]
                        if name in self.next_sub_parameters
                        else self.next_sub_buffers[name]
                    )
                    restore_tensor_storage(target, cpu_shard, device="cpu")


def _get_submodules(block: nn.Module) -> dict[str, nn.Module]:
    """Extract attn, mlp, adaln_proj sub-modules from a DiT block.

    Returns a dict mapping sub-module name to the module instance.
    Only sub-modules with significant weight are included; norm/RoPE
    stay resident via move_non_block_state_to_device.
    """
    subs: dict[str, nn.Module] = {}
    if hasattr(block, "adaln_proj"):
        subs["adaln_proj"] = block.adaln_proj
    if hasattr(block, "attn"):
        subs["attn"] = block.attn
    if hasattr(block, "mlp"):
        subs["mlp"] = block.mlp
    return subs


def _submodule_param_dict(module: nn.Module) -> dict[str, nn.Parameter]:
    """Collect parameters of a sub-module, prefixed by sub-module-relative name."""
    return dict(module.named_parameters())


def _submodule_buffer_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return dict(module.named_buffers())


class SubmoduleOffloadBackend:
    """Backend that installs sub-module level offload hooks on DiT blocks.

    Reuses DLO's copy_stream, comm_stream, pin_memory, AllGather, and
    mmap infrastructure.  Only the DiT blocks get sub-module hooks;
    Token Refiner blocks keep their existing DLO block-level hooks.

    The circular ring spans all 50 blocks × 3 sub-modules = 150 hooks:
        adaln_0 → attn_0 → mlp_0 → adaln_1 → ... → mlp_49 → adaln_0
    """

    _HOOK_NAME = "submodule_offload"

    def __init__(
        self,
        config: Any,
        device: torch.device,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.enabled = False
        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        self._hooks: list[SubmoduleOffloadHook] = []
        self._blocks: list[nn.Module] = []

        self._dp_group: torch.distributed.ProcessGroup | None = None
        self._dp_size: int = getattr(config, "dp_size", 1)
        self._rank: int = 0
        self._pin_memory: bool = getattr(config, "pin_cpu_memory", True)

        self._max_sub_params: dict[torch.dtype, int] = {}
        self._all_hook_groups: list[list[SubmoduleOffloadHook]] = []

        self._shared_gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        self._shared_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]

        self._init_process_group()

    def is_enabled(self) -> bool:
        return self.enabled

    def shutdown(self) -> None:
        self.disable()

    def _init_process_group(self) -> None:
        if self._dp_size <= 1:
            logger.info("Sub-module offload: dp_size=1, running without AllGather")
            self._dp_group = None
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Sub-module offload with dp_size > 1 requires an initialized process group."
            )

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_world_size,
            get_dp_group,
        )

        if get_data_parallel_world_size() > 1:
            self._dp_group = get_dp_group()
        else:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_sequence_parallel_world_size,
                get_sp_group,
            )
            sp_size = get_sequence_parallel_world_size()
            if sp_size and sp_size > 1:
                self._dp_group = get_sp_group()
                self._dp_size = sp_size
            else:
                self._dp_group = None
                self._dp_size = 1

        from vllm_omni.diffusion.distributed.parallel_state import get_data_parallel_rank
        self._rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0

    def _allocate_shared_buffers(self) -> None:
        """Allocate 2 shared GPU buffers sized to the largest sub-module."""
        for dtype, max_numel in self._max_sub_params.items():
            if self._dp_size > 1:
                padded = max_numel
            else:
                padded = max_numel
            for slot in range(2):
                if self._shared_gpu_buffers[slot] is None:
                    self._shared_gpu_buffers[slot] = {}
                self._shared_gpu_buffers[slot][dtype] = torch.empty(
                    padded, dtype=dtype, device=self.device
                )

        if self._dp_size > 1:
            for dtype, max_numel in self._max_sub_params.items():
                shard_sz = (max_numel + self._dp_size - 1) // self._dp_size
                for slot in range(2):
                    if self._shared_shard_buffers[slot] is None:
                        self._shared_shard_buffers[slot] = {}
                    self._shared_shard_buffers[slot][dtype] = torch.empty(
                        shard_sz, dtype=dtype, device=self.device
                    )

    def _compute_max_sub_params(self, blocks: list[nn.Module]) -> None:
        """Find the maximum sub-module parameter count across all blocks."""
        for block in blocks:
            subs = _get_submodules(block)
            for sub_name, sub_module in subs.items():
                params = _submodule_param_dict(sub_module)
                bufs = _submodule_buffer_dict(sub_module)
                grouped = group_named_tensors_by_dtype(params, bufs)
                for dtype, named_weights in grouped.items():
                    specs = describe_tensor_storage(named_weights, None)
                    total_numel = sum(spec.storage_numel for spec in specs)
                    self._max_sub_params[dtype] = max(
                        self._max_sub_params.get(dtype, 0), total_numel
                    )

    def _install_hook_ring(
        self,
        blocks: list[nn.Module],
    ) -> list[SubmoduleOffloadHook]:
        """Install a circular ring of sub-module hooks across all blocks.

        Ring order: adaln_0 → attn_0 → mlp_0 → adaln_1 → ... → mlp_49 → adaln_0
        """
        sub_sequence: list[tuple[str, nn.Module]] = []
        for block in blocks:
            subs = _get_submodules(block)
            if "adaln_proj" in subs:
                sub_sequence.append(("adaln_proj", subs["adaln_proj"]))
            if "attn" in subs:
                sub_sequence.append(("attn", subs["attn"]))
            if "mlp" in subs:
                sub_sequence.append(("mlp", subs["mlp"]))

        if len(sub_sequence) <= 1:
            raise ValueError("Sub-module offload requires at least 2 sub-modules")

        hooks: list[SubmoduleOffloadHook] = []
        n = len(sub_sequence)

        for i, (sub_name, sub_module) in enumerate(sub_sequence):
            next_idx = (i + 1) % n
            prev_idx = (i - 1) % n
            next_name, next_module = sub_sequence[next_idx]
            prev_name, prev_module = sub_sequence[prev_idx]

            hook = SubmoduleOffloadHook(
                sub_module=sub_module,
                sub_name=sub_name,
                device=self.device,
                dp_group=self._dp_group,
                dp_size=self._dp_size,
                rank=self._rank,
                copy_stream=self.copy_stream,
                comm_stream=self.comm_stream,
                pin_memory=self._pin_memory,
                shared_gpu_buffers=self._shared_gpu_buffers,
                shared_shard_buffers=self._shared_shard_buffers,
            )

            hook.next_sub_parameters = _submodule_param_dict(next_module)
            hook.next_sub_buffers = _submodule_buffer_dict(next_module)
            hook._materialization_probe = module_materialization_probe(next_module)

            hook._shard_and_pin(
                params=hook.next_sub_parameters,
                bufs=hook.next_sub_buffers,
            )

            hooks.append(hook)

        for i, hook in enumerate(hooks):
            hook._prev_hook = hooks[(i - 1) % n]
            hook.next_hook = hooks[(i + 1) % n]
            hook.current_slot = i % 2

        hooks[0]._is_first_in_ring = True

        return hooks

    def enable(self, pipeline: nn.Module) -> None:
        """Install sub-module hooks on DiT blocks and keep non-block state resident."""
        from vllm_omni.diffusion.offloader.plan_resolver import (
            resolve_offload_plan,
        )
        from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan

        plan = get_offload_plan(pipeline)
        resolved = resolve_offload_plan(pipeline, plan, self.config)

        dit_components = [c for c in resolved.dits if c.selected and c.stacks]
        if not dit_components:
            raise ValueError("No DiT component found for sub-module offload")

        all_hooks: list[SubmoduleOffloadHook] = []

        for dit_comp in dit_components:
            for stack in dit_comp.stacks:
                blocks = list(stack.blocks)
                if len(blocks) <= 1:
                    continue

                self._compute_max_sub_params(blocks)
                self._allocate_shared_buffers()

                hooks = self._install_hook_ring(blocks)
                self._all_hook_groups.append(hooks)
                all_hooks.extend(hooks)
                self._blocks.extend(blocks)

                for hook in hooks:
                    hook._allocate_device_buffers()

                for block in blocks:
                    subs = _get_submodules(block)
                    for sub_module in subs.values():
                        move_non_block_state_to_device(sub_module, (sub_module,), self.device)

            if dit_comp.children:
                for child in dit_comp.children:
                    if child.selected and child.stacks:
                        self._offload_nested_submodule(child)
                    else:
                        from vllm_omni.diffusion.offloader.component_utils import prepare_component
                        prepare_component(
                            child.module,
                            child.path,
                            device=self.device,
                            stage_on_demand=True,
                            blockwise=False,
                            staged_components=getattr(self, "_staged_components", set()),
                        )

            move_non_block_state_to_device(dit_comp.module, [s.blocks for s in dit_comp.stacks], self.device)

        first_hook = all_hooks[0] if all_hooks else None
        if first_hook is not None and first_hook._prev_hook is not None:
            first_hook._prev_hook.prefetch_layer(first_hook.current_slot, non_blocking=False)
            first_hook._prev_hook._wait_for_weights(first_hook.current_slot)

        for hook in all_hooks:
            hook.attach()

        self._hooks = all_hooks
        logger.info(
            "Sub-module offload enabled: %d hooks across %d blocks, "
            "dp_size=%d, pin_memory=%s",
            len(all_hooks),
            len(self._blocks),
            self._dp_size,
            self._pin_memory,
        )

    def _offload_nested_submodule(self, component: Any) -> None:
        """Stream a nested DiT submodule (e.g. token_refiner) via DLO block hooks."""
        stack = component.stacks[0]
        logger.info(
            "Sub-module offload: nested submodule '%s' using DLO block hooks (%d blocks)",
            component.path,
            len(stack.blocks),
        )
        move_non_block_state_to_device(component.module, (stack.blocks,), self.device)

        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
            DistributedLayerwiseOffloadBackend,
        )
        dlo_like = DistributedLayerwiseOffloadBackend.__new__(DistributedLayerwiseOffloadBackend)
        dlo_like.config = self.config
        dlo_like.device = self.device
        dlo_like.copy_stream = self.copy_stream
        dlo_like.comm_stream = self.comm_stream
        dlo_like.dp_group = self._dp_group
        dlo_like.dp_size = self._dp_size
        dlo_like.rank = self._rank
        dlo_like.pin_memory = self._pin_memory
        dlo_like._using_mmap = False
        dlo_like._using_rank_local_mmap = False
        dlo_like._using_registered_mmap = False
        dlo_like._all_hook_groups = []
        dlo_like._blocks = []
        dlo_like._encoder_modules = []
        dlo_like._staged_components = set()
        dlo_like._component_transports = {}
        dlo_like._init_process_group = lambda: None
        dlo_like._allocate_shared_buffers = lambda: None
        dlo_like._install_hook_group(stack.blocks, DIT_COMPONENT, use_dit_mmap=False)

    def disable(self) -> None:
        """Remove all hooks and restore weights to CPU."""
        for hook in self._hooks:
            hook.detach()
            hook.restore_to_cpu()
        self._hooks.clear()
        self._all_hook_groups.clear()
        self._blocks.clear()
        self._shared_gpu_buffers = [None, None]
        self._shared_shard_buffers = [None, None]

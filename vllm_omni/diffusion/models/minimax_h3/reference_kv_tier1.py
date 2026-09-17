# SPDX-License-Identifier: Apache-2.0
"""Host-resident reference-KV staging for MiniMax-H3.

MiniMax-H3 uses bidirectional packed self-attention, so visual-condition K/V
changes after the first DiT block as target tokens evolve. This module provides
two explicit experimental caches backed by one contiguous pinned-host tensor:
Tier1 bulk-stages every layer before a reuse forward, while Tier2 uses a small
device ring and prefetches the next layer on a private copy stream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.profiler import record_function
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

_log = init_logger(__name__)

_ENABLE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1"
_TIER2_ENABLE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2"
_REFRESH_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_REFRESH_INTERVAL"
_RING_SIZE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_RING_SIZE"


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _refresh_interval_from_env() -> int:
    raw = os.getenv(_REFRESH_ENV, "2").strip()
    try:
        interval = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_REFRESH_ENV} must be an integer, got {raw!r}") from exc
    if interval < 0:
        raise ValueError(f"{_REFRESH_ENV} must be >= 0, got {interval}")
    return interval


def _ring_size_from_env() -> int:
    raw = os.getenv(_RING_SIZE_ENV, "2").strip()
    try:
        ring_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_RING_SIZE_ENV} must be an integer, got {raw!r}") from exc
    if not 2 <= ring_size <= 3:
        raise ValueError(f"{_RING_SIZE_ENV} must be 2 or 3, got {ring_size}")
    return ring_size


@dataclass
class MiniMaxH3ReferenceKVTier1Stats:
    refresh_steps: int = 0
    h2d_steps: int = 0
    captured_layers: int = 0
    substituted_layers: int = 0
    h2d_bytes: int = 0


@dataclass
class MiniMaxH3ReferenceKVTier2Stats(MiniMaxH3ReferenceKVTier1Stats):
    prefetched_layers: int = 0
    ready_wait_layers: int = 0
    max_device_staging_bytes: int = 0


class MiniMaxH3ReferenceKVTier1State:
    """Request-local pinned-host K/V pool plus one-forward device staging.

    ``refresh_interval=2`` captures on even denoise steps and reuses on odd
    steps.  ``refresh_interval=0`` captures only step 0, matching a strict
    populate-once prototype; this is intentionally explicit because full-depth
    MiniMax-H3 reference K/V is not invariant across denoise steps.
    """

    tier_name = "Tier1"

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        refresh_interval: int,
        pin_memory: bool = True,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if global_reference_rows <= 0:
            raise ValueError(
                "global_reference_rows must be positive, "
                f"got {global_reference_rows}"
            )
        if refresh_interval < 0:
            raise ValueError(
                f"refresh_interval must be >= 0, got {refresh_interval}"
            )
        self.num_layers = int(num_layers)
        self.global_reference_rows = int(global_reference_rows)
        self.device = torch.device(device)
        self.refresh_interval = int(refresh_interval)
        self.pin_memory = bool(pin_memory)
        self.stats = MiniMaxH3ReferenceKVTier1Stats()

        self._host_pool: torch.Tensor | None = None
        self._device_pool: torch.Tensor | None = None
        self._local_reference_positions: torch.Tensor | None = None
        self._local_sequence_rows: int | None = None
        self._step: int | None = None
        self._refresh = False
        self._inactive = False
        self._layers_seen: set[int] = set()

    @classmethod
    def from_environment(
        cls,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
    ) -> MiniMaxH3ReferenceKVTier1State | None:
        tier2_enabled = _env_enabled(_TIER2_ENABLE_ENV)
        tier1_enabled = _env_enabled(_ENABLE_ENV)
        if not tier1_enabled and not tier2_enabled:
            return None
        tier_name = "Tier2" if tier2_enabled else "Tier1"
        if global_reference_rows <= 0:
            _log.info(
                "MiniMax-H3 reference-KV %s skipped: request has no visual "
                "condition rows (T2VA has no reference KV to offload)",
                tier_name,
            )
            return None
        interval = _refresh_interval_from_env()
        if interval == 0:
            _log.warning(
                "MiniMax-H3 reference-KV %s strict populate-once mode is "
                "enabled. Full-depth reference K/V drifts in bidirectional "
                "attention, so this mode is lossy; use refresh interval 2 for "
                "the measured quality-oriented prototype.",
                tier_name,
            )
        if tier2_enabled:
            state = MiniMaxH3ReferenceKVTier2State(
                num_layers=num_layers,
                global_reference_rows=global_reference_rows,
                device=device,
                refresh_interval=interval,
                ring_size=_ring_size_from_env(),
            )
        else:
            state = cls(
                num_layers=num_layers,
                global_reference_rows=global_reference_rows,
                device=device,
                refresh_interval=interval,
            )
        _log.info(
            "MiniMax-H3 reference-KV %s enabled: layers=%d, global_ref_rows=%d, "
            "refresh_interval=%d, device=%s%s",
            tier_name,
            num_layers,
            global_reference_rows,
            interval,
            device,
            f", ring_size={state.ring_size}" if tier2_enabled else "",
        )
        return state

    @property
    def host_pool(self) -> torch.Tensor | None:
        return self._host_pool

    @property
    def device_pool(self) -> torch.Tensor | None:
        return self._device_pool

    @property
    def host_bytes(self) -> int:
        if self._host_pool is None:
            return 0
        return self._host_pool.numel() * self._host_pool.element_size()

    @property
    def current_step(self) -> int | None:
        return self._step

    def _is_refresh_step(self, step: int) -> bool:
        if self._inactive:
            return False
        if self._host_pool is None:
            return True
        return self.refresh_interval > 0 and step % self.refresh_interval == 0

    def begin_step(self, step: int) -> None:
        if self._step is not None:
            raise RuntimeError(
                f"reference-KV Tier1 step {self._step} is still active"
            )
        self._step = int(step)
        self._layers_seen.clear()
        self._refresh = self._is_refresh_step(step)
        if self._inactive:
            return
        if self._refresh:
            self.stats.refresh_steps += 1
            return
        if self._host_pool is None:
            raise RuntimeError("reference-KV Tier1 reuse requested before populate")
        with record_function("minimax_h3.ref_kv_tier1.h2d_bulk"):
            self._device_pool = self._host_pool.to(
                device=self.device,
                non_blocking=False,
            )
        copied = self.host_bytes
        self.stats.h2d_steps += 1
        self.stats.h2d_bytes += copied
        _log.info(
            "MiniMax-H3 reference-KV Tier1 H2D bulk: step=%d, bytes=%d, layers=%d",
            step,
            copied,
            self.num_layers,
        )

    def set_local_reference_mask(self, reference_mask: torch.Tensor) -> None:
        if reference_mask.dim() != 1 or reference_mask.dtype != torch.bool:
            raise ValueError(
                "reference_mask must be a 1-D bool tensor, got "
                f"shape={tuple(reference_mask.shape)} dtype={reference_mask.dtype}"
            )
        positions = torch.nonzero(reference_mask, as_tuple=False).view(-1)
        if self._local_reference_positions is None:
            self._local_reference_positions = positions
            self._local_sequence_rows = reference_mask.numel()
            self._inactive = positions.numel() == 0
            if self._inactive and self._refresh and self.stats.refresh_steps:
                self.stats.refresh_steps -= 1
                self._refresh = False
            return
        if (
            self._local_sequence_rows != reference_mask.numel()
            or self._local_reference_positions.numel() != positions.numel()
        ):
            raise RuntimeError(
                "MiniMax-H3 reference row layout changed within one denoise request"
            )

    def _allocate_host_pool(self, k: torch.Tensor) -> None:
        if self._local_reference_positions is None:
            raise RuntimeError("local reference mask must be set before KV capture")
        shape = (
            2,
            self.num_layers,
            self._local_reference_positions.numel(),
            *k.shape[1:],
        )
        kwargs = {
            "size": shape,
            "dtype": k.dtype,
            "device": "cpu",
        }
        try:
            self._host_pool = torch.empty(
                **kwargs,
                pin_memory=self.pin_memory,
            )
        except RuntimeError:
            if self.device.type != "cpu" or not self.pin_memory:
                raise
            # CPU-only unit-test environments may not provide a pin allocator.
            self._host_pool = torch.empty(**kwargs)
        _log.info(
            "MiniMax-H3 reference-KV %s host pool populated lazily: "
            "shape=%s, pinned=%s, bytes=%d",
            self.tier_name,
            tuple(self._host_pool.shape),
            self._host_pool.is_pinned(),
            self.host_bytes,
        )

    def process_layer(
        self,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._step is None:
            raise RuntimeError("begin_step must be called before processing KV")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index {layer_index} outside [0, {self.num_layers})"
            )
        if layer_index in self._layers_seen:
            raise RuntimeError(
                f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}"
            )
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} != {v.shape}")
        positions = self._local_reference_positions
        if positions is None:
            raise RuntimeError("local reference mask was not initialized")
        if self._local_sequence_rows != k.shape[0]:
            raise ValueError(
                "local reference mask/KV row mismatch: "
                f"{self._local_sequence_rows} != {k.shape[0]}"
            )
        if k.shape[0] == 0 or positions.numel() == 0:
            self._layers_seen.add(layer_index)
            return k, v

        if self._refresh:
            if self._host_pool is None:
                self._allocate_host_pool(k)
            assert self._host_pool is not None
            expected = tuple(self._host_pool.shape[2:])
            actual = (positions.numel(), *k.shape[1:])
            if actual != expected:
                raise ValueError(
                    f"reference K/V shape changed: expected {expected}, got {actual}"
                )
            with record_function(
                f"minimax_h3.ref_kv_{self.tier_name.lower()}.populate_host"
            ):
                self._host_pool[0, layer_index].copy_(
                    k.index_select(0, positions),
                    non_blocking=False,
                )
                self._host_pool[1, layer_index].copy_(
                    v.index_select(0, positions),
                    non_blocking=False,
                )
            self.stats.captured_layers += 1
        else:
            if self._device_pool is None:
                raise RuntimeError("bulk H2D staging is missing on a reuse step")
            k.index_copy_(0, positions, self._device_pool[0, layer_index])
            v.index_copy_(0, positions, self._device_pool[1, layer_index])
            self.stats.substituted_layers += 1
        self._layers_seen.add(layer_index)
        return k, v

    def end_step(self) -> None:
        if self._step is None:
            raise RuntimeError("no active reference-KV Tier1 step")
        if len(self._layers_seen) != self.num_layers:
            raise RuntimeError(
                "MiniMax-H3 reference-KV Tier1 saw "
                f"{len(self._layers_seen)}/{self.num_layers} layers in step {self._step}"
            )
        # The authoritative cache remains on pinned host memory.  Releasing the
        # only device-pool reference returns the full-L staging allocation to the
        # accelerator allocator after every reuse forward.
        self._device_pool = None
        self._step = None
        self._layers_seen.clear()

    def abort_step(self) -> None:
        self._device_pool = None
        self._step = None
        self._layers_seen.clear()

    def close(self) -> None:
        if self._step is not None:
            self.abort_step()
        self._device_pool = None
        _log.info(
            "MiniMax-H3 reference-KV Tier1 summary: host_bytes=%d, "
            "refresh_steps=%d, h2d_steps=%d, captured_layers=%d, "
            "substituted_layers=%d, h2d_bytes=%d",
            self.host_bytes,
            self.stats.refresh_steps,
            self.stats.h2d_steps,
            self.stats.captured_layers,
            self.stats.substituted_layers,
            self.stats.h2d_bytes,
        )
        self._host_pool = None
        self._local_reference_positions = None
        self._local_sequence_rows = None


class MiniMaxH3ReferenceKVTier2State(MiniMaxH3ReferenceKVTier1State):
    """Layerwise asynchronous H2D staging backed by a small device ring.

    Refresh steps retain Tier1's synchronous per-layer D2H capture. Reuse steps
    prefetch one layer at a time from the pinned host pool. Layer ``i + 1`` is
    submitted immediately after layer ``i`` consumes its slot, so its H2D can
    overlap the current layer's attention and MLP work.
    """

    tier_name = "Tier2"

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        refresh_interval: int,
        ring_size: int = 2,
        pin_memory: bool = True,
    ) -> None:
        if not 2 <= ring_size <= 3:
            raise ValueError(f"ring_size must be 2 or 3, got {ring_size}")
        super().__init__(
            num_layers=num_layers,
            global_reference_rows=global_reference_rows,
            device=device,
            refresh_interval=refresh_interval,
            pin_memory=pin_memory,
        )
        self.ring_size = int(ring_size)
        self.stats = MiniMaxH3ReferenceKVTier2Stats()
        self._device_ring: torch.Tensor | None = None
        self._copy_stream: Any | None = None
        self._slot_layers: list[int | None] = [None] * self.ring_size
        self._ready_events: list[Any | None] = [None] * self.ring_size
        self._consumed_events: list[Any | None] = [None] * self.ring_size

    @property
    def device_pool(self) -> torch.Tensor | None:
        return self._device_ring

    @property
    def device_staging_bytes(self) -> int:
        if self._device_ring is None:
            return 0
        return self._device_ring.numel() * self._device_ring.element_size()

    def _ensure_device_ring(self) -> None:
        if self._device_ring is not None:
            return
        if self._host_pool is None:
            raise RuntimeError("Tier2 ring allocation requested before populate")
        shape = (self.ring_size, 2, *self._host_pool.shape[2:])
        self._device_ring = torch.empty(
            shape,
            dtype=self._host_pool.dtype,
            device=self.device,
        )
        if self.device.type != "cpu":
            try:
                self._copy_stream = current_omni_platform.Stream()
            except Exception as exc:
                _log.warning(
                    "MiniMax-H3 reference-KV Tier2 could not create an "
                    "asynchronous copy stream; falling back to synchronous "
                    "layerwise H2D: %s",
                    exc,
                )
                self._copy_stream = None
        self.stats.max_device_staging_bytes = max(
            self.stats.max_device_staging_bytes,
            self.device_staging_bytes,
        )
        _log.info(
            "MiniMax-H3 reference-KV Tier2 device ring allocated: "
            "shape=%s, bytes=%d, ring_size=%d",
            tuple(self._device_ring.shape),
            self.device_staging_bytes,
            self.ring_size,
        )

    def _prefetch_layer(self, layer_index: int) -> None:
        if self._host_pool is None or self._device_ring is None:
            raise RuntimeError("Tier2 prefetch requested before pool allocation")
        slot = layer_index % self.ring_size
        consumed = self._consumed_events[slot]
        host_layer = self._host_pool[:, layer_index]
        device_slot = self._device_ring[slot]

        with record_function("minimax_h3.ref_kv_tier2.prefetch"):
            if self._copy_stream is None:
                device_slot.copy_(host_layer, non_blocking=False)
                ready = None
            else:
                try:
                    if consumed is not None:
                        self._copy_stream.wait_event(consumed)
                    ready = current_omni_platform.Event()
                    with current_omni_platform.stream(self._copy_stream):
                        device_slot.copy_(
                            host_layer,
                            non_blocking=host_layer.is_pinned(),
                        )
                        ready.record(self._copy_stream)
                except Exception as exc:
                    _log.warning(
                        "MiniMax-H3 reference-KV Tier2 asynchronous prefetch "
                        "failed at layer %d; falling back to synchronous "
                        "layerwise H2D: %s",
                        layer_index,
                        exc,
                    )
                    try:
                        self._copy_stream.synchronize()
                    except Exception:
                        pass
                    self._copy_stream = None
                    device_slot.copy_(host_layer, non_blocking=False)
                    ready = None

        self._slot_layers[slot] = layer_index
        self._ready_events[slot] = ready
        self.stats.prefetched_layers += 1
        self.stats.h2d_bytes += host_layer.numel() * host_layer.element_size()

    def begin_step(self, step: int) -> None:
        if self._step is not None:
            raise RuntimeError(
                f"reference-KV Tier2 step {self._step} is still active"
            )
        self._step = int(step)
        self._layers_seen.clear()
        self._refresh = self._is_refresh_step(step)
        if self._inactive:
            return
        if self._refresh:
            self.stats.refresh_steps += 1
            return
        if self._host_pool is None:
            raise RuntimeError("reference-KV Tier2 reuse requested before populate")

        self._ensure_device_ring()
        self.stats.h2d_steps += 1
        self._prefetch_layer(0)
        _log.info(
            "MiniMax-H3 reference-KV Tier2 layerwise H2D: "
            "step=%d, layer_bytes=%d, layers=%d, ring_size=%d",
            step,
            self.host_bytes // self.num_layers,
            self.num_layers,
            self.ring_size,
        )

    def process_layer(
        self,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._refresh:
            return super().process_layer(layer_index, k, v)
        if self._step is None:
            raise RuntimeError("begin_step must be called before processing KV")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index {layer_index} outside [0, {self.num_layers})"
            )
        if layer_index in self._layers_seen:
            raise RuntimeError(
                f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}"
            )
        if layer_index != len(self._layers_seen):
            raise RuntimeError(
                "Tier2 layerwise prefetch requires sequential layer execution: "
                f"expected {len(self._layers_seen)}, got {layer_index}"
            )
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} != {v.shape}")
        positions = self._local_reference_positions
        if positions is None:
            raise RuntimeError("local reference mask was not initialized")
        if self._local_sequence_rows != k.shape[0]:
            raise ValueError(
                "local reference mask/KV row mismatch: "
                f"{self._local_sequence_rows} != {k.shape[0]}"
            )
        if self._inactive or positions.numel() == 0:
            self._layers_seen.add(layer_index)
            return k, v
        if self._device_ring is None:
            raise RuntimeError("Tier2 device ring is missing on a reuse step")

        slot = layer_index % self.ring_size
        if self._slot_layers[slot] != layer_index:
            raise RuntimeError(
                f"Tier2 slot {slot} contains layer {self._slot_layers[slot]}, "
                f"expected {layer_index}"
            )
        ready = self._ready_events[slot]
        compute_stream = None
        if ready is not None:
            compute_stream = current_omni_platform.current_stream()
            with record_function("minimax_h3.ref_kv_tier2.wait"):
                compute_stream.wait_event(ready)
            self.stats.ready_wait_layers += 1

        with record_function("minimax_h3.ref_kv_tier2.substitute"):
            k.index_copy_(0, positions, self._device_ring[slot, 0])
            v.index_copy_(0, positions, self._device_ring[slot, 1])
        self.stats.substituted_layers += 1
        self._layers_seen.add(layer_index)

        if compute_stream is not None:
            consumed = current_omni_platform.Event()
            consumed.record(compute_stream)
            self._consumed_events[slot] = consumed
        else:
            self._consumed_events[slot] = None

        next_layer = layer_index + 1
        if next_layer < self.num_layers:
            self._prefetch_layer(next_layer)
        return k, v

    def end_step(self) -> None:
        if self._step is None:
            raise RuntimeError("no active reference-KV Tier2 step")
        if len(self._layers_seen) != self.num_layers:
            raise RuntimeError(
                "MiniMax-H3 reference-KV Tier2 saw "
                f"{len(self._layers_seen)}/{self.num_layers} layers in step {self._step}"
            )
        self._step = None
        self._layers_seen.clear()

    def abort_step(self) -> None:
        if self._copy_stream is not None:
            try:
                self._copy_stream.synchronize()
            except Exception:
                _log.debug("Tier2 copy-stream cleanup failed", exc_info=True)
        self._device_ring = None
        self._copy_stream = None
        self._slot_layers = [None] * self.ring_size
        self._ready_events = [None] * self.ring_size
        self._consumed_events = [None] * self.ring_size
        self._step = None
        self._layers_seen.clear()

    def close(self) -> None:
        if self._step is not None:
            self.abort_step()
        elif self._copy_stream is not None:
            try:
                self._copy_stream.synchronize()
            except Exception:
                _log.debug("Tier2 copy-stream close failed", exc_info=True)
        _log.info(
            "MiniMax-H3 reference-KV Tier2 summary: host_bytes=%d, "
            "device_staging_bytes=%d, refresh_steps=%d, h2d_steps=%d, "
            "captured_layers=%d, substituted_layers=%d, "
            "prefetched_layers=%d, ready_wait_layers=%d, h2d_bytes=%d, "
            "inactive=%s",
            self.host_bytes,
            self.device_staging_bytes,
            self.stats.refresh_steps,
            self.stats.h2d_steps,
            self.stats.captured_layers,
            self.stats.substituted_layers,
            self.stats.prefetched_layers,
            self.stats.ready_wait_layers,
            self.stats.h2d_bytes,
            self._inactive,
        )
        self._device_ring = None
        self._copy_stream = None
        self._slot_layers = [None] * self.ring_size
        self._ready_events = [None] * self.ring_size
        self._consumed_events = [None] * self.ring_size
        self._host_pool = None
        self._local_reference_positions = None
        self._local_sequence_rows = None


__all__ = [
    "MiniMaxH3ReferenceKVTier1State",
    "MiniMaxH3ReferenceKVTier1Stats",
    "MiniMaxH3ReferenceKVTier2State",
    "MiniMaxH3ReferenceKVTier2Stats",
]

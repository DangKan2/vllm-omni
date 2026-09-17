# SPDX-License-Identifier: Apache-2.0
"""Host-resident reference-KV staging for MiniMax-H3.

MiniMax-H3 uses bidirectional packed self-attention, so visual-condition K/V
changes after the first DiT block as target tokens evolve.  This module provides
an explicit experimental cache: refresh steps capture visual-condition K/V into
one contiguous pinned-host tensor, and reuse steps synchronously bulk-stage the
whole tensor once before the DiT forward.  The device staging tensor is released
at the end of that forward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.profiler import record_function
from vllm.logger import init_logger

_log = init_logger(__name__)

_ENABLE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1"
_REFRESH_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_REFRESH_INTERVAL"


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


@dataclass
class MiniMaxH3ReferenceKVTier1Stats:
    refresh_steps: int = 0
    h2d_steps: int = 0
    captured_layers: int = 0
    substituted_layers: int = 0
    h2d_bytes: int = 0


class MiniMaxH3ReferenceKVTier1State:
    """Request-local pinned-host K/V pool plus one-forward device staging.

    ``refresh_interval=2`` captures on even denoise steps and reuses on odd
    steps.  ``refresh_interval=0`` captures only step 0, matching a strict
    populate-once prototype; this is intentionally explicit because full-depth
    MiniMax-H3 reference K/V is not invariant across denoise steps.
    """

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
        self._layers_seen: set[int] = set()

    @classmethod
    def from_environment(
        cls,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
    ) -> MiniMaxH3ReferenceKVTier1State | None:
        if not _env_enabled(_ENABLE_ENV):
            return None
        if global_reference_rows <= 0:
            _log.info(
                "MiniMax-H3 reference-KV Tier1 skipped: request has no visual "
                "condition rows (T2VA has no reference KV to offload)"
            )
            return None
        interval = _refresh_interval_from_env()
        if interval == 0:
            _log.warning(
                "MiniMax-H3 reference-KV Tier1 strict populate-once mode is "
                "enabled. Full-depth reference K/V drifts in bidirectional "
                "attention, so this mode is lossy; use refresh interval 2 for "
                "the measured quality-oriented prototype."
            )
        state = cls(
            num_layers=num_layers,
            global_reference_rows=global_reference_rows,
            device=device,
            refresh_interval=interval,
        )
        _log.info(
            "MiniMax-H3 reference-KV Tier1 enabled: layers=%d, global_ref_rows=%d, "
            "refresh_interval=%d, device=%s",
            num_layers,
            global_reference_rows,
            interval,
            device,
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
            "MiniMax-H3 reference-KV Tier1 host pool populated lazily: "
            "shape=%s, pinned=%s, bytes=%d",
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
            with record_function("minimax_h3.ref_kv_tier1.populate_host"):
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


__all__ = [
    "MiniMaxH3ReferenceKVTier1State",
    "MiniMaxH3ReferenceKVTier1Stats",
]

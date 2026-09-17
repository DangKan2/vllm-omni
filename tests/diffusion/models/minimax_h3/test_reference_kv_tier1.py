# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
    MiniMaxH3ReferenceKVTier1State,
    MiniMaxH3ReferenceKVTier2State,
)


def _kv(layer: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
    return values + base + layer * 100, values + base + layer * 100 + 50


def test_tier1_populates_host_and_bulk_stages_once_per_reuse_step() -> None:
    state = MiniMaxH3ReferenceKVTier1State(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        pin_memory=False,
    )
    mask = torch.tensor([False, True, False, True])

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    state.begin_step(0)
    state.set_local_reference_mask(mask)
    for layer in range(2):
        k, v = _kv(layer, 0)
        captured.append((k[mask].clone(), v[mask].clone()))
        out_k, out_v = state.process_layer(layer, k, v)
        assert out_k.data_ptr() == k.data_ptr()
        assert out_v.data_ptr() == v.data_ptr()
    state.end_step()

    assert state.host_pool is not None
    assert state.host_pool.is_contiguous()
    assert tuple(state.host_pool.shape) == (2, 2, 2, 2, 3)
    assert state.device_pool is None

    state.begin_step(1)
    assert state.device_pool is not None
    for layer in range(2):
        k, v = _kv(layer, 1000)
        current_k = k.clone()
        current_v = v.clone()
        out_k, out_v = state.process_layer(layer, k, v)
        torch.testing.assert_close(out_k[mask], captured[layer][0])
        torch.testing.assert_close(out_v[mask], captured[layer][1])
        torch.testing.assert_close(out_k[~mask], current_k[~mask])
        torch.testing.assert_close(out_v[~mask], current_v[~mask])
    state.end_step()

    assert state.device_pool is None
    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 1
    assert state.stats.captured_layers == 2
    assert state.stats.substituted_layers == 2
    assert state.stats.h2d_bytes == state.host_bytes


def test_tier1_interval_zero_is_strict_populate_once() -> None:
    state = MiniMaxH3ReferenceKVTier1State(
        num_layers=1,
        global_reference_rows=1,
        device=torch.device("cpu"),
        refresh_interval=0,
        pin_memory=False,
    )
    mask = torch.tensor([True, False])

    state.begin_step(0)
    state.set_local_reference_mask(mask)
    k0 = torch.tensor([[[1.0]], [[2.0]]])
    v0 = torch.tensor([[[3.0]], [[4.0]]])
    state.process_layer(0, k0, v0)
    state.end_step()

    for step in (1, 2):
        state.begin_step(step)
        k = torch.full_like(k0, 10.0 + step)
        v = torch.full_like(v0, 20.0 + step)
        state.process_layer(0, k, v)
        assert k[0].item() == 1.0
        assert v[0].item() == 3.0
        state.end_step()

    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 2
    assert state.stats.captured_layers == 1
    assert state.stats.substituted_layers == 2


def test_tier2_layerwise_ring_reuses_cached_reference_rows() -> None:
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=4,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=2,
        pin_memory=False,
    )
    mask = torch.tensor([False, True, False, True])

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    state.begin_step(0)
    state.set_local_reference_mask(mask)
    for layer in range(4):
        k, v = _kv(layer, 0)
        captured.append((k[mask].clone(), v[mask].clone()))
        state.process_layer(layer, k, v)
    state.end_step()

    state.begin_step(1)
    for layer in range(4):
        k, v = _kv(layer, 1000)
        current_k = k.clone()
        current_v = v.clone()
        out_k, out_v = state.process_layer(layer, k, v)
        torch.testing.assert_close(out_k[mask], captured[layer][0])
        torch.testing.assert_close(out_v[mask], captured[layer][1])
        torch.testing.assert_close(out_k[~mask], current_k[~mask])
        torch.testing.assert_close(out_v[~mask], current_v[~mask])
    state.end_step()

    assert state.device_pool is not None
    assert tuple(state.device_pool.shape) == (2, 2, 2, 2, 3)
    assert state.device_staging_bytes == state.host_bytes // 2
    assert state.stats.refresh_steps == 1
    assert state.stats.h2d_steps == 1
    assert state.stats.captured_layers == 4
    assert state.stats.substituted_layers == 4
    assert state.stats.prefetched_layers == 4
    assert state.stats.h2d_bytes == state.host_bytes
    assert state.stats.max_device_staging_bytes == state.device_staging_bytes


def test_tier2_zero_reference_rank_is_inactive() -> None:
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=2,
        pin_memory=False,
    )
    mask = torch.zeros(4, dtype=torch.bool)

    for step in range(2):
        state.begin_step(step)
        state.set_local_reference_mask(mask)
        for layer in range(2):
            k, v = _kv(layer, float(step))
            state.process_layer(layer, k, v)
        state.end_step()

    assert state.host_pool is None
    assert state.device_pool is None
    assert state.stats.refresh_steps == 0
    assert state.stats.h2d_steps == 0
    assert state.stats.captured_layers == 0
    assert state.stats.substituted_layers == 0
    assert state.stats.prefetched_layers == 0


def test_environment_selects_tier2(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1", raising=False)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_RING_SIZE", "3")
    state = MiniMaxH3ReferenceKVTier1State.from_environment(
        num_layers=4,
        global_reference_rows=2,
        device=torch.device("cpu"),
    )

    assert isinstance(state, MiniMaxH3ReferenceKVTier2State)
    assert state.ring_size == 3

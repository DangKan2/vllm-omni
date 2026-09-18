# SPDX-License-Identifier: Apache-2.0

import json

import torch

from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
    MiniMaxH3ReferenceKVObserverState,
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
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER", raising=False)
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


def test_observer_measures_multiple_intervals_without_substitution(tmp_path) -> None:
    state = MiniMaxH3ReferenceKVObserverState(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        intervals=(2, 3),
        sample_rows=2,
        sample_heads=1,
        output_path=str(tmp_path / "drift.jsonl"),
    )
    mask = torch.tensor([False, True, False, True])

    for step in range(3):
        state.begin_step(step, sigma=1.0 - 0.1 * step)
        state.set_local_reference_mask(mask)
        for layer in range(2):
            k, v = _kv(layer, float(step * 10))
            expected_k = k.clone()
            expected_v = v.clone()
            out_k, out_v = state.process_layer(layer, k, v)
            assert out_k.data_ptr() == k.data_ptr()
            assert out_v.data_ptr() == v.data_ptr()
            torch.testing.assert_close(out_k, expected_k)
            torch.testing.assert_close(out_v, expected_v)
        state.end_step()

    assert state.stats.observed_steps == 3
    assert state.stats.sampled_layers == 6
    assert state.stats.metric_records == 12
    state.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "drift.cpu.jsonl").read_text().splitlines()
    ]
    assert [record["kind"] for record in records].count("run_start") == 1
    assert [record["kind"] for record in records].count("layout") == 1
    assert [record["kind"] for record in records].count("run_end") == 1

    metrics = [record for record in records if record["kind"] == "metric"]
    assert len(metrics) == 12
    step_one = [record for record in metrics if record["step"] == 1]
    assert {(record["interval"], record["source_step"], record["age"])
            for record in step_one} == {(2, 0, 1), (3, 0, 1)}
    step_two = [record for record in metrics if record["step"] == 2]
    assert {(record["interval"], record["source_step"], record["age"])
            for record in step_two} == {(3, 0, 2)}
    assert {record["kv"] for record in metrics} == {"K", "V"}
    assert all(record["rel_l2"] > 0 for record in metrics)
    assert all(record["abs_error_max"] > 0 for record in metrics)
    assert all(record["source_sigma"] == 1.0 for record in metrics)
    assert all(record["sigma_delta"] < 0 for record in metrics)


def test_observer_zero_reference_rank_is_inactive(tmp_path) -> None:
    state = MiniMaxH3ReferenceKVObserverState(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
        intervals=(2,),
        sample_rows=2,
        sample_heads=1,
        output_path=str(tmp_path / "inactive.jsonl"),
    )
    mask = torch.zeros(4, dtype=torch.bool)

    for step in range(2):
        state.begin_step(step, sigma=float(1 - step))
        state.set_local_reference_mask(mask)
        for layer in range(2):
            k, v = _kv(layer, float(step))
            out_k, out_v = state.process_layer(layer, k, v)
            assert out_k.data_ptr() == k.data_ptr()
            assert out_v.data_ptr() == v.data_ptr()
        state.end_step()

    assert state.host_bytes == 0
    assert state.stats.observed_steps == 0
    assert state.stats.sampled_layers == 0
    assert state.stats.metric_records == 0
    state.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "inactive.cpu.jsonl").read_text().splitlines()
    ]
    end = next(record for record in records if record["kind"] == "run_end")
    assert end["inactive"] is True
    assert end["metric_records"] == 0


def test_environment_observer_takes_precedence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_REFRESH_INTERVAL", "invalid")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_INTERVALS", "4,2")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_ROWS", "3")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_HEADS", "1")
    monkeypatch.setenv(
        "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_OUTPUT",
        str(tmp_path / "observer.jsonl"),
    )

    state = MiniMaxH3ReferenceKVTier1State.from_environment(
        num_layers=2,
        global_reference_rows=2,
        device=torch.device("cpu"),
    )

    assert isinstance(state, MiniMaxH3ReferenceKVObserverState)
    assert state.intervals == (2, 4)
    assert state.sample_rows == 3
    assert state.sample_heads == 1
    state.close()

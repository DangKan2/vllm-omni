# SPDX-License-Identifier: Apache-2.0
"""Test actual Cache-DiT decisions, refresh bounds, and single-step replay."""

import importlib.util
from pathlib import Path

import cache_dit
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.reference_kv_cachedit import (
    configure_reference_cache_schedule,
)
from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
    MiniMaxH3ReferenceKVTier1State,
    MiniMaxH3ReferenceKVTier2State,
    ReferenceKVRefreshRequired,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


_spec = importlib.util.spec_from_file_location(
    "joint_schedule_test_helpers", Path(__file__).with_name("test_reference_kv_cachedit.py")
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def joint_state(model, *, interval=4, maximum=7, quant="none", device="cpu"):
    state = helpers.state_for(MiniMaxH3ReferenceKVTier2State, interval=interval, quant=quant, device=device)
    state.cache_dit_policy = "hit"
    state.max_refresh_interval = maximum
    return state


def run_joint(model, state, steps):
    mask = torch.tensor([True, False, True, False], device=state.device)
    state.set_global_reference_mask(mask)
    model._context_manager.set_context(model._context_names[0])
    context = model._context_manager.get_context()
    rows = []
    for step in range(steps):
        state.begin_step(step)
        state.set_local_reference_mask(mask)
        probe = False
        try:
            compact = state.should_skip_reference_projection
            output = model(
                torch.zeros(2 if compact else 4, 3, device=state.device),
                reference_kv_tier1_state=state,
                reference_kv_compact=compact,
            )
        except ReferenceKVRefreshRequired:
            probe = True
        if probe:
            state.promote_step_to_refresh()
            output = model(
                torch.zeros(4, 3, device=state.device),
                reference_kv_tier1_state=state,
                reference_kv_compact=False,
            )
        torch.testing.assert_close(output, torch.full_like(output, 0.21), atol=1e-5, rtol=1e-5)
        rows.append(
            {
                "step": step,
                "probe": probe,
                "refresh": state.is_reference_refresh_step,
                "reason": state.refresh_reason,
                "age": state.reference_age,
                "version": state.reference_epoch,
                "hit": context.get_current_step() in context.cached_steps,
            }
        )
        assert state.cache_dit_schedule_active
        state.end_step()
    return rows


@pytest.mark.parametrize("bn", [0, 1])
@pytest.mark.parametrize("taylor", [False, True])
@pytest.mark.parametrize("quant", ["none", "int8"])
def test_due_hits_defer_until_actual_miss_without_double_clock(bn, taylor, quant):
    model = helpers.install(helpers.TinyTransformer(), bn=bn, taylor=taylor, static=True, steps=12)
    physical = list(model.blocks)
    model._context_manager.set_context(model._context_names[0])
    context = model._context_manager.get_context()
    context.cache_config.steps_computation_mask = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    state = joint_state(model, quant=quant)
    rows = run_joint(model, state, 12)
    assert [row["step"] for row in rows if row["refresh"]] == [0, 5, 10]
    assert [row["step"] for row in rows if row["probe"]] == [5, 10]
    assert rows[4]["hit"] and rows[4]["reason"] == "cache_hit_deferred"
    assert rows[9]["hit"] and rows[9]["version"] == 1
    assert rows[10]["version"] == 2
    assert context.executed_steps == context.transformer_executed_steps == 12
    assert physical[0].calls.count(5) == 2  # compact probe + full refresh
    assert physical[2].calls.count(5) == 1  # no duplicate middle execution
    assert state.stats.captured_layers == 18
    # A new request must get its own version/age and Cache-DiT clock.
    cache_dit.refresh_context(model, num_inference_steps=12, verbose=False)
    next_state = joint_state(model, quant=quant)
    next_rows = run_joint(model, next_state, 12)
    assert next_rows[0]["version"] == 0
    assert context.executed_steps == 12
    state.close()
    next_state.close()


@pytest.mark.parametrize("bn", [0, 1])
@pytest.mark.parametrize("taylor", [False, True])
def test_hard_age_limit_overrides_otherwise_static_hits(bn, taylor):
    model = helpers.install(helpers.TinyTransformer(), bn=bn, taylor=taylor, static=True, steps=16)
    state = joint_state(model)
    rows = run_joint(model, state, 16)
    assert [row["step"] for row in rows if row["refresh"]] == [0, 7, 14]
    assert [row["step"] for row in rows if row["reason"] == "maximum_age"] == [7, 14]
    assert not any(row["probe"] for row in rows)
    assert all(row["age"] is None or row["age"] <= 7 for row in rows)
    assert all(not row["hit"] for row in rows if row["refresh"])
    assert state.stats.captured_layers == 18
    state.close()


def test_unsupported_wrapper_and_tier_keep_refresh_priority():
    model = helpers.TinyTransformer()
    state = helpers.state_for(MiniMaxH3ReferenceKVTier2State)
    state.cache_dit_policy = "hit"
    state.max_refresh_interval = 7
    configure_reference_cache_schedule(model.blocks, state)
    assert not state.cache_dit_schedule_active
    helpers.run_steps(model, state, 8)
    assert state.stats.refresh_steps == 2
    wrapped = helpers.install(helpers.TinyTransformer(), static=True)
    tier1 = helpers.state_for(MiniMaxH3ReferenceKVTier1State)
    tier1.cache_dit_policy = "hit"
    tier1.max_refresh_interval = 7
    configure_reference_cache_schedule(wrapped.blocks, tier1)
    assert not tier1.cache_dit_schedule_active
    helpers.run_steps(wrapped, tier1, 8)
    assert tier1.stats.refresh_steps == 2
    state.close()
    tier1.close()


def test_interval_one_can_refresh_on_layout_forced_miss():
    model = helpers.install(helpers.TinyTransformer(), taylor=True, static=True, steps=4)
    state = joint_state(model, interval=1, maximum=2)
    rows = run_joint(model, state, 4)
    assert [row["step"] for row in rows if row["refresh"]] == [0, 1, 2, 3]
    assert model._context_manager.get_context().executed_steps == 4
    state.close()


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("natural_miss", [False, True])
def test_npu_int8_probe_refresh_and_ring_reuse(natural_miss):
    model = helpers.install(helpers.TinyTransformer(), taylor=True, static=True, steps=10)
    model._context_manager.set_context(model._context_names[0])
    context = model._context_manager.get_context()
    if natural_miss:
        context.cache_config.steps_computation_mask = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    state = joint_state(model, quant="int8", device="npu:0")
    rows = run_joint(model, state, 10)
    torch.npu.synchronize()
    assert context.executed_steps == 10
    assert [row["step"] for row in rows if row["refresh"]] == [0, 5 if natural_miss else 7]
    assert any(row["probe"] for row in rows) == natural_miss
    assert state.stats.captured_layers == 12
    assert state.effective_host_quantization == "int8"
    assert state.host_pool.is_pinned()
    assert state._copy_stream is not None
    state.close()

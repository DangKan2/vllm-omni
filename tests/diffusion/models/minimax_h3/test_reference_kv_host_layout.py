# SPDX-License-Identifier: Apache-2.0
"""Layer ownership, scale alignment and reuse stay correct with a contiguous pool."""

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
    MiniMaxH3ReferenceKVTier2State,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("quantization", ["none", "int8", "fp8"])
@pytest.mark.parametrize("ring_size", [2, 3])
def test_layer_major_host_pool_preserves_versions_and_skipped_layers(quantization, ring_size):
    # L != 2 catches accidentally retaining the old [K/V, L, ...] indexing.
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=5,
        global_reference_rows=3,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=ring_size,
        pin_memory=False,
        skip_reference_projection=True,
        host_quantization=quantization,
    )
    mask = torch.tensor([True, False, True, False, False, True, False])
    expected = {}
    for step in range(6):
        state.begin_step(step)
        state.set_global_reference_mask(mask)
        state.set_local_reference_mask(torch.zeros(4, dtype=torch.bool))
        if step % 2 == 0:
            for layer in range(5):
                key = torch.arange(42).reshape(1, 7, 2, 3).float() + 2 + step * 20 + layer * 5
                value = -key * 0.5 - 7
                state.process_post_parallel_layer(layer, key, value, compact=False, parallel_strategy="ulysses")
                expected[layer] = (key[0, mask].clone(), value[0, mask].clone())
            assert state.host_pool is not None
            assert state.host_pool.shape == (5, 2, 3, 2, 3)
            assert all(state.host_pool[layer].is_contiguous() for layer in range(5))
            if quantization == "int8":
                assert state._host_scale_pool.shape == (5, 2, 2)
                assert all(state._host_scale_pool[layer].is_contiguous() for layer in range(5))
        else:
            layers = [0, 1, 4] if step == 3 else range(5)
            for layer in layers:
                if step == 3 and layer == 4:
                    state.mark_block_cache_skipped([2, 3])
                target_key = torch.full((1, 4, 2, 3), 1000.0 + step + layer)
                target_value = torch.full_like(target_key, -2000.0 - step - layer)
                key, value = state.process_post_parallel_layer(
                    layer,
                    target_key,
                    target_value,
                    compact=True,
                    parallel_strategy="ulysses",
                )
                tolerance = {"none": (0, 0), "fp8": (0.06, 0.1)}.get(quantization)
                if quantization == "int8":
                    # Symmetric INT8 rounding has at most half a quantization
                    # step of error; small values need an absolute bound.
                    maximum = max(tensor.abs().max().item() for tensor in expected[layer])
                    tolerance = (0, maximum / 254 + 1e-5)
                torch.testing.assert_close(key[0, :3], expected[layer][0], rtol=tolerance[0], atol=tolerance[1])
                torch.testing.assert_close(value[0, :3], expected[layer][1], rtol=tolerance[0], atol=tolerance[1])
                torch.testing.assert_close(key[:, 3:], target_key, rtol=0, atol=0)
                torch.testing.assert_close(value[:, 3:], target_value, rtol=0, atol=0)
            if quantization == "int8":
                assert state._device_scale_ring.shape == (ring_size, 2, 2)
        state.end_step()
    assert state.stats.refresh_steps == 3
    assert state.stats.captured_layers == 15
    assert state.stats.substituted_layers == 13
    assert state.stats.block_cached_layers == 2
    state.close()


def test_legacy_tier2_host_views_keep_original_indexing():
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=3,
        global_reference_rows=2,
        device=torch.device("cpu"),
        refresh_interval=2,
        ring_size=2,
        pin_memory=False,
    )
    mask = torch.tensor([False, True, False, True])
    reference = {}
    state.begin_step(0)
    state.set_local_reference_mask(mask)
    for layer in range(3):
        key = torch.arange(24).reshape(4, 2, 3).float() + layer * 100
        value = key + 50
        reference[layer] = (key[mask].clone(), value[mask].clone())
        state.process_layer(layer, key, value)
    state.end_step()
    assert state.host_pool.shape == (2, 3, 2, 2, 3)
    assert not state._host_pool_layer_major
    state.begin_step(1)
    for layer in range(3):
        key, value = state.process_layer(layer, torch.zeros(4, 2, 3), torch.zeros(4, 2, 3))
        torch.testing.assert_close(key[mask], reference[layer][0], rtol=0, atol=0)
        torch.testing.assert_close(value[mask], reference[layer][1], rtol=0, atol=0)
    state.end_step()
    state.close()

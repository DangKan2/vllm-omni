# SPDX-License-Identifier: Apache-2.0
"""Real cache and model entrypoint contracts, without NPU/weights imports."""

import ast
import importlib.util
import logging
import sys
from types import SimpleNamespace

import pytest
import torch
from tests.diffusion.models.minimax_h3.test_reference_isolation import MODEL, ROOT, load_cpu_cache_module, load_function


def config_module(monkeypatch):
    name = "vllm_omni.diffusion.kv_offload_config"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_omni/diffusion/kv_offload_config.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def isolated_factory():
    path = MODEL / "reference_isolation_cache.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    namespace = {
        "torch": torch,
        "init_logger": logging.getLogger,
        "MiniMaxH3ReferenceKVTier2State": load_cpu_cache_module().MiniMaxH3ReferenceKVTier2State,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(tree), str(path), "exec"), namespace)
    return namespace["create_isolated_reference_cache"]


def test_standalone_isolate_reuses_host_int8_two_slot_cache_once():
    state = isolated_factory()(num_layers=3, global_reference_rows=2, device=torch.device("cpu"))
    assert state.post_parallel_cache and state.refresh_interval == 0 and state.ring_size == 2
    assert state.host_quantization == "int8" and state.skip_reference_projection
    assert state.cache_dit_policy == "refresh" and not state.cache_dit_schedule_active
    mask = torch.tensor([False, True, False, True, False, False])
    original = torch.randn(1, 6, 2, 4)
    for step in range(3):
        state.begin_step(step)
        state.set_global_reference_mask(mask)
        state.set_local_reference_mask(mask)
        for layer in range(3):
            like = original if step == 0 else original[:, ~mask].clone()
            k, v = state.process_post_parallel_layer(layer, like, like, compact=step > 0, parallel_strategy="ulysses")
            assert k.shape[1] == 6 and v.shape == k.shape
        state.end_step()
    assert state.host_pool.device.type == "cpu" and state.host_pool.dtype == torch.int8
    assert state._device_ring.shape[0] == 2
    assert state.stats.refresh_steps == 1 and state.stats.captured_layers == 3
    assert state.stats.substituted_layers == 6 and state.stats.prefetched_layers > 0
    state.close()
    assert state.host_pool is None and state._device_ring is None


def test_isolate_without_reference_has_no_cache():
    assert isolated_factory()(num_layers=3, global_reference_rows=0, device=torch.device("cpu")) is None


def test_split_metadata_preserves_backend_quantization_policy():
    from tests.diffusion.models.minimax_h3.test_reference_isolation import Metadata, make_state

    state = make_state()
    metadata = Metadata(extra={"laser_input_scale": 256.0, "kv_cache_dtype": "float"})
    result = state._dense_metadata(metadata, torch.randn(1, 5, 2, 4), 7)
    assert result.extra["kv_cache_dtype"] == "float" and result.extra["laser_input_scale"] == 256.0
    assert metadata.extra == {"laser_input_scale": 256.0, "kv_cache_dtype": "float"}


@pytest.mark.parametrize("enabled", [False, True])
def test_model_entrypoint_priority_over_periodic_policy(enabled, monkeypatch):
    config_module(monkeypatch)
    calls = []
    value = {
        "mode": "tier2",
        "kv_refresh_interval": 16,
        "skip_reference_projection": True,
        "cache_dit_policy": "hit",
        "max_refresh_interval": 32,
    }

    class Existing:
        @staticmethod
        def from_config(**kwargs):
            calls.append(("periodic", kwargs["config"]))
            return SimpleNamespace()

    def isolate(**kwargs):
        calls.append(("isolate", kwargs))
        return SimpleNamespace()

    from tests.diffusion.models.minimax_h3.test_reference_isolation import isolation_module

    fn = load_function(
        MODEL / "minimax_h3_transformer.py",
        "create_reference_kv_tier1_state",
        {
            "reference_isolation_enabled": isolation_module.reference_isolation_enabled,
            "create_isolated_reference_cache": isolate,
            "MiniMaxH3ReferenceKVTier1State": Existing,
            "iter_minimax_h3_blocks": lambda b: iter(b),
            "configure_reference_cache_schedule": lambda *a: None,
        },
    )
    model = SimpleNamespace(od_config=SimpleNamespace(kv_isolate=enabled, kv_offload_config=value), blocks=[1, 2, 3])
    fn(model, global_reference_rows=2, device=torch.device("cpu"))
    assert calls[0][0] == ("isolate" if enabled else "periodic")
    if enabled:
        assert "config" not in calls[0][1]
    else:
        assert calls[0][1].kv_refresh_interval == 16 and calls[0][1].cache_dit_policy == "hit"
    assert value["kv_refresh_interval"] == 16 and value["cache_dit_policy"] == "hit"


def test_no_flags_preserves_early_exit_without_layer_traversal(monkeypatch):
    config_module(monkeypatch)
    from tests.diffusion.models.minimax_h3.test_reference_isolation import isolation_module

    fn = load_function(
        MODEL / "minimax_h3_transformer.py",
        "create_reference_kv_tier1_state",
        {"reference_isolation_enabled": isolation_module.reference_isolation_enabled},
    )
    model = SimpleNamespace(od_config=SimpleNamespace(kv_offload_config=None))
    assert fn(model, global_reference_rows=2, device=torch.device("cpu")) is None


def test_isolate_alone_and_both_flags_call_identical_factory(monkeypatch):
    config_module(monkeypatch)
    from tests.diffusion.models.minimax_h3.test_reference_isolation import isolation_module

    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return None

    fn = load_function(
        MODEL / "minimax_h3_transformer.py",
        "create_reference_kv_tier1_state",
        {
            "reference_isolation_enabled": isolation_module.reference_isolation_enabled,
            "create_isolated_reference_cache": factory,
            "iter_minimax_h3_blocks": lambda b: iter(b),
        },
    )
    for value in [
        None,
        {"mode": "tier1", "kv_refresh_interval": 16},
        {
            "mode": "tier2",
            "kv_refresh_interval": 8,
            "kv_host_quantization": "none",
            "skip_reference_projection": True,
            "cache_dit_policy": "hit",
            "max_refresh_interval": 32,
        },
    ]:
        model = SimpleNamespace(od_config=SimpleNamespace(kv_isolate=True, kv_offload_config=value), blocks=[1, 2])
        fn(model, global_reference_rows=2, device=torch.device("cpu"))
    assert calls[0] == calls[1] == calls[2]


def test_offload_config_validation_unchanged(monkeypatch):
    config = config_module(monkeypatch)
    with pytest.raises(ValueError):
        config.parse_reference_kv_config({"mode": "tier2", "kv_refresh_interval": -1})
    for value in ["true", 1, {}, []]:
        with pytest.raises(ValueError, match="boolean"):
            config.validate_kv_isolate(value)
    for value in [None, False, True]:
        config.validate_kv_isolate(value)

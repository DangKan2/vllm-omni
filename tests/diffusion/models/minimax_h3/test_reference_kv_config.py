# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate serving opt-in, scheduling and default-path isolation."""

from dataclasses import asdict, fields
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.kv_offload_config import parse_reference_kv_config
from vllm_omni.diffusion.models.minimax_h3.reference_kv_tier1 import (
    MiniMaxH3ReferenceKVTier1State,
    MiniMaxH3ReferenceKVTier2State,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("mode,quant", [("tier1", "none"), ("tier2", "none"), ("tier2", "int8"), ("tier2", "fp8")])
def test_public_config_selects_requested_cache(mode, quant):
    config = parse_reference_kv_config(
        {
            "mode": mode,
            "kv_refresh_interval": 16,
            "kv_host_quantization": quant,
            "skip_reference_projection": True,
        }
    )
    state = MiniMaxH3ReferenceKVTier1State.from_config(
        config=config,
        num_layers=5,
        global_reference_rows=3,
        device=torch.device("cpu"),
    )
    assert isinstance(state, MiniMaxH3ReferenceKVTier2State) == (mode == "tier2")
    assert state.refresh_interval == 16
    assert state.host_quantization == quant
    assert state.cache_dit_policy == "refresh"
    if mode == "tier2":
        assert state.ring_size == 2
    state.close()


@pytest.mark.parametrize(
    "updates",
    [
        {"kv_refresh_interval": True},
        {"kv_refresh_interval": -1},
        {"mode": "unknown"},
        {"kv_host_quantization": "unknown"},
        {"skip_reference_projection": 1},
        {"unknown": 1},
        {"cache_dit_policy": "unknown"},
        {"mode": "tier1", "kv_host_quantization": "int8"},
        {"mode": "tier1", "cache_dit_policy": "hit"},
        {"kv_refresh_interval": 0, "cache_dit_policy": "hit"},
        {"skip_reference_projection": False, "cache_dit_policy": "hit"},
        {"cache_dit_policy": "hit", "max_refresh_interval": 15},
        {"cache_dit_policy": "hit", "max_refresh_interval": True},
        {"cache_dit_policy": "hit", "max_refresh_interval": "32"},
        {"max_refresh_interval": 32},
    ],
)
def test_invalid_config_fails_before_worker_start(updates):
    value = {"mode": "tier2", "kv_refresh_interval": 16, "skip_reference_projection": True}
    value.update(updates)
    with pytest.raises(ValueError):
        parse_reference_kv_config(value)


def test_hit_policy_has_one_bounded_default():
    value = {"mode": "tier2", "kv_refresh_interval": 16, "skip_reference_projection": True, "cache_dit_policy": "hit"}
    config = parse_reference_kv_config(value)
    assert config.max_refresh_interval == 32
    value["max_refresh_interval"] = 48
    assert parse_reference_kv_config(value).max_refresh_interval == 48
    assert parse_reference_kv_config(asdict(config)) == config


def test_unconfigured_path_ignores_all_legacy_environment(monkeypatch):
    for name, value in {
        "TIER1": "1",
        "TIER2": "1",
        "OBSERVER": "1",
        "SKIP_PROJECTION": "1",
        "REFRESH_INTERVAL": "invalid",
        "RING_SIZE": "50",
        "HOST_QUANTIZATION": "int8",
        "CACHE_DIT_POLICY": "hit",
        "MAX_REFRESH_INTERVAL": "32",
    }.items():
        monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_" + name, value)
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    class MustNotInspectBlocks:
        def __iter__(self):
            raise AssertionError("Disabled feature traversed model blocks")

    model = SimpleNamespace(od_config=SimpleNamespace(kv_offload_config=None), blocks=MustNotInspectBlocks())
    assert parse_reference_kv_config(None) is None
    assert (
        MiniMaxH3DiTModel.create_reference_kv_tier1_state(
            model,
            global_reference_rows=3,
            device=torch.device("cpu"),
        )
        is None
    )
    assert MiniMaxH3DiTModel._cache_dit_adapter_config.cached_adapter_cls is None


def test_config_survives_stage_and_worker_projection():
    from vllm_omni.config.omni_config import _DiffusionConfigProjection
    from vllm_omni.config.stage_config import StageDeployConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    value = {
        "mode": "tier2",
        "kv_refresh_interval": 16,
        "kv_host_quantization": "int8",
        "skip_reference_projection": True,
        "cache_dit_policy": "hit",
        "max_refresh_interval": 32,
    }
    assert "kv_offload_config" in {f.name for f in fields(StageDeployConfig)}
    projected = _DiffusionConfigProjection.from_kwargs(kv_offload_config=value)
    assert projected.kv_offload_config == value
    worker = OmniDiffusionConfig.from_kwargs(kv_offload_config=value)
    assert worker.kv_offload_config == value

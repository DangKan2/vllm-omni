# SPDX-License-Identifier: Apache-2.0
"""Actual v0.31 stage-to-worker configuration and priority integration."""

import dataclasses

import pytest


def test_cli_stage_and_worker_keep_isolation_option():
    from vllm_omni.config.omni_config import _DiffusionConfigProjection
    from vllm_omni.config.stage_config import StageDeployConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    assert "kv_isolate" in {f.name for f in dataclasses.fields(StageDeployConfig)}
    stage = StageDeployConfig(stage_id=0, kv_isolate=True)
    value = dataclasses.asdict(stage)
    projected = _DiffusionConfigProjection.from_kwargs(kv_isolate=value["kv_isolate"])
    assert projected.kv_isolate is True
    worker = OmniDiffusionConfig.from_kwargs(kv_isolate=projected.kv_isolate)
    assert worker.kv_isolate is True
    assert OmniDiffusionConfig.from_kwargs().kv_isolate is False
    assert _DiffusionConfigProjection.from_kwargs().kv_isolate is False


@pytest.mark.parametrize("enabled", [False, True])
def test_config_validation_follows_isolate_priority(enabled):
    from vllm_omni.config.omni_config import _DiffusionConfigProjection
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    ignored = {"mode": "tier2", "kv_refresh_interval": -1}
    for cls in [_DiffusionConfigProjection, OmniDiffusionConfig]:
        if enabled:
            value = cls.from_kwargs(kv_isolate=True, kv_offload_config=ignored)
            assert value.kv_isolate is True and value.kv_offload_config == ignored
        else:
            with pytest.raises(ValueError):
                cls.from_kwargs(kv_offload_config=ignored)


@pytest.mark.parametrize("value", ["true", 1, {}, []])
def test_nonboolean_isolation_config_fails_early(value):
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    with pytest.raises(ValueError, match="boolean"):
        _DiffusionConfigProjection.from_kwargs(kv_isolate=value)

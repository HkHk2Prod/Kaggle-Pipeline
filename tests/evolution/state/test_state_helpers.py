"""Unit tests for the state_io module-level helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kaggle_pipeline.evolution.ecosystem.serialization import EcosystemSerializer
from kaggle_pipeline.evolution.ecosystem.state import PIPELINE_VERSION, EcosystemState
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.state_io import (
    check_pipeline_version,
    collect_resume_states,
    pick_resume_serializer,
)


def _state(registry, settings, *, batch_index=0, version=PIPELINE_VERSION):
    state = EcosystemState(
        config_snapshot={},
        batch_index=batch_index,
        registry=registry,
        population=ModelPopulation(settings),
        oof_store=OOFStore(),
        rng_state=np.random.default_rng(0).bit_generator.state,
    )
    state.pipeline_version = version
    return state


def test_check_pipeline_version_returns_none_on_match(registry, settings):
    assert check_pipeline_version(_state(registry, settings), strict=False) is None


def test_check_pipeline_version_returns_message_when_mismatched(registry, settings):
    msg = check_pipeline_version(_state(registry, settings, version="0.0.0-fake"), strict=False)
    assert msg and "0.0.0-fake" in msg


def test_check_pipeline_version_raises_when_strict(registry, settings):
    with pytest.raises(ValueError, match="pipeline_version"):
        check_pipeline_version(_state(registry, settings, version="0.0.0-fake"), strict=True)


def test_pick_resume_serializer_prefers_local_when_it_has_data(tmp_path, registry, settings):
    local = EcosystemSerializer(tmp_path / "live", keep_last_n=3)
    local.save(_state(registry, settings, batch_index=1), reason="bootstrap")
    cfg = SimpleNamespace(
        previous_state_dir=str(tmp_path / "previous"),
        state_dir=str(tmp_path / "live"),
        keep_last_n_checkpoints=3,
    )
    chosen = pick_resume_serializer(local, cfg)
    assert chosen is local


def test_pick_resume_serializer_returns_none_when_nothing_exists(tmp_path):
    empty = EcosystemSerializer(tmp_path / "live", keep_last_n=3)
    cfg = SimpleNamespace(
        previous_state_dir=None,
        state_dir=str(tmp_path / "live"),
        keep_last_n_checkpoints=3,
    )
    assert pick_resume_serializer(empty, cfg) is None


def test_pick_resume_serializer_falls_back_to_previous_when_local_empty(
    tmp_path, registry, settings
):
    previous = tmp_path / "previous"
    EcosystemSerializer(previous, keep_last_n=3).save(
        _state(registry, settings, batch_index=5), reason="prior"
    )
    local = EcosystemSerializer(tmp_path / "live", keep_last_n=3)
    cfg = SimpleNamespace(
        previous_state_dir=str(previous),
        state_dir=str(tmp_path / "live"),
        keep_last_n_checkpoints=3,
    )
    chosen = pick_resume_serializer(local, cfg)
    assert chosen is not None
    assert chosen is not local
    assert chosen.load().batch_index == 5


def test_collect_resume_states_prefers_local(tmp_path, registry, settings):
    local = EcosystemSerializer(tmp_path / "live", keep_last_n=3)
    local.save(_state(registry, settings, batch_index=4), reason="bootstrap")
    cfg = SimpleNamespace(
        previous_state_dir=None,
        state_dir=str(tmp_path / "live"),
        keep_last_n_checkpoints=3,
    )
    states = collect_resume_states(local, cfg)
    assert len(states) == 1
    assert states[0].batch_index == 4


def test_collect_resume_states_loads_explicit_previous(tmp_path, registry, settings):
    previous = tmp_path / "previous"
    EcosystemSerializer(previous, keep_last_n=3).save(
        _state(registry, settings, batch_index=9), reason="prior"
    )
    local = EcosystemSerializer(tmp_path / "live", keep_last_n=3)
    cfg = SimpleNamespace(
        previous_state_dir=str(previous),
        state_dir=str(tmp_path / "live"),
        keep_last_n_checkpoints=3,
    )
    states = collect_resume_states(local, cfg)
    assert len(states) == 1
    assert states[0].batch_index == 9


def test_collect_resume_states_empty_when_nothing(tmp_path):
    local = EcosystemSerializer(tmp_path / "live", keep_last_n=3)
    cfg = SimpleNamespace(
        previous_state_dir=None,
        state_dir=str(tmp_path / "live"),
        keep_last_n_checkpoints=3,
    )
    assert collect_resume_states(local, cfg) == []

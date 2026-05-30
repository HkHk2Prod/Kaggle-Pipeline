"""Ecosystem checkpointing: atomic save, manifest, rotation, and load roundtrip."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.ecosystem.serialization import EcosystemSerializer
from kaggle_pipeline.evolution.ecosystem.state import EcosystemState
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.models.registry import ModelPopulation


def _state(registry, settings, *, batch_index=0):
    return EcosystemState(
        config_snapshot={"seed": 0, "max_active_features": settings.max_active_features},
        batch_index=batch_index,
        registry=registry,
        population=ModelPopulation(settings),
        oof_store=OOFStore(),
        rng_state=np.random.default_rng(0).bit_generator.state,
    )


def test_state_roundtrip_preserves_features(tmp_path, registry, settings):
    serializer = EcosystemSerializer(tmp_path / "st", keep_last_n=5)
    serializer.save(_state(registry, settings, batch_index=3), reason="test", summary={"x": 1})

    loaded = serializer.load()
    assert loaded.batch_index == 3
    assert len(loaded.registry.all_features()) == len(registry.all_features())
    # The restored registry is usable: its materializer lock was re-created.
    assert loaded.registry.materializer is not None


def test_manifest_records_metadata(tmp_path, registry, settings):
    serializer = EcosystemSerializer(tmp_path / "st")
    serializer.save(_state(registry, settings, batch_index=7), reason="batch_complete")
    manifest = serializer.read_manifest()
    assert manifest["batch_index"] == 7
    assert manifest["feature_count"] == len(registry.all_features())
    assert manifest["notes"] == "batch_complete"
    assert "config_hash" in manifest


def test_checkpoint_rotation_keeps_last_n(tmp_path, registry, settings):
    serializer = EcosystemSerializer(tmp_path / "st", keep_last_n=3)
    for i in range(5):
        serializer.save(_state(registry, settings, batch_index=i))
    kept = sorted((tmp_path / "st" / "checkpoints").glob("checkpoint_*"))
    assert len(kept) == 3
    # The latest pointer still resolves and loads.
    assert serializer.load().batch_index == 4

"""Saving and loading the ecosystem state to disk.

Each checkpoint is a directory holding a pickled :class:`EcosystemState` plus
JSON sidecars (manifest, config, summary) for auditability. Writes are atomic
(write to a temp dir, then ``os.replace`` into place) and old checkpoints are
pruned to ``keep_last_n``. A ``latest.json`` pointer records the most recent
checkpoint so ``load()`` needs no argument.

The full state goes through pickle (it carries numpy arrays and nested genome
objects that have no clean JSON form); the JSON sidecars stay human-readable and
make a checkpoint inspectable without unpickling.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

from kaggle_pipeline.evolution.ecosystem.state import EcosystemState
from kaggle_pipeline.evolution.storage.hashing import stable_hash

STATE_PICKLE = "ecosystem_state.pkl"
MANIFEST = "manifest.json"
CONFIG = "config.json"
SUMMARY = "summary.json"
LATEST_POINTER = "latest.json"


class EcosystemSerializer:
    """Writes/reads versioned ecosystem checkpoints under a state directory."""

    def __init__(self, state_dir: str | Path, *, keep_last_n: int = 5, atomic: bool = True):
        self.state_dir = Path(state_dir)
        self.checkpoints_dir = self.state_dir / "checkpoints"
        self.keep_last_n = keep_last_n
        self.atomic = atomic

    # --- save ---------------------------------------------------------------
    def save(
        self,
        state: EcosystemState,
        *,
        reason: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = self._next_checkpoint_id()
        final_dir = self.checkpoints_dir / f"checkpoint_{checkpoint_id:06d}"
        tmp_dir = self.checkpoints_dir / f".tmp_{checkpoint_id:06d}_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with (tmp_dir / STATE_PICKLE).open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        _write_json(tmp_dir / CONFIG, state.config_snapshot)
        if summary is not None:
            _write_json(tmp_dir / SUMMARY, summary)
        _write_json(tmp_dir / MANIFEST, self._manifest(state, checkpoint_id, reason, summary))

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)  # atomic directory rename
        self._update_latest(final_dir, checkpoint_id)
        self._prune_old_checkpoints()
        return final_dir

    def _manifest(
        self, state: EcosystemState, checkpoint_id: int, reason: str | None, summary: dict | None
    ) -> dict[str, Any]:
        models = state.population
        best = models.absolute_score_ranking()
        best_genome = best[0] if best else None
        return {
            "checkpoint_id": checkpoint_id,
            "created_at": time.time(),
            "batch_index": state.batch_index,
            "pipeline_version": state.pipeline_version,
            "python_version": state.python_version,
            "config_hash": stable_hash(state.config_snapshot)[:16],
            "feature_count": len(state.registry.all_features()),
            "model_count": len(models.all_genomes()),
            "best_score": (
                best_genome.score_set.score if best_genome and best_genome.score_set else None
            ),
            "best_model_id": best_genome.model_id if best_genome else None,
            "ensemble_available": bool(state.ensemble_state),
            "random_seed": state.config_snapshot.get("seed"),
            "notes": reason or "",
        }

    def _update_latest(self, final_dir: Path, checkpoint_id: int) -> None:
        _write_json(
            self.state_dir / LATEST_POINTER,
            {"checkpoint_id": checkpoint_id, "path": str(final_dir.resolve())},
        )

    def _next_checkpoint_id(self) -> int:
        existing = sorted(self.checkpoints_dir.glob("checkpoint_*"))
        if not existing:
            return 1
        return int(existing[-1].name.split("_")[1]) + 1

    def _prune_old_checkpoints(self) -> None:
        existing = sorted(self.checkpoints_dir.glob("checkpoint_*"))
        for old in existing[: -self.keep_last_n] if self.keep_last_n > 0 else []:
            shutil.rmtree(old, ignore_errors=True)

    # --- load ---------------------------------------------------------------
    def latest_path(self) -> Path | None:
        pointer = self.state_dir / LATEST_POINTER
        if pointer.exists():
            data = json.loads(pointer.read_text(encoding="utf-8"))
            path = Path(data["path"])
            if (path / STATE_PICKLE).exists():
                return path
        existing = (
            sorted(self.checkpoints_dir.glob("checkpoint_*"))
            if self.checkpoints_dir.exists()
            else []
        )
        return existing[-1] if existing else None

    def load(self, path: str | Path | None = None) -> EcosystemState:
        target = Path(path) if path is not None else self.latest_path()
        if target is None or not (Path(target) / STATE_PICKLE).exists():
            raise FileNotFoundError(f"no ecosystem checkpoint found under {self.state_dir}")
        with (Path(target) / STATE_PICKLE).open("rb") as fh:
            state = pickle.load(fh)
        if not isinstance(state, EcosystemState):
            raise ValueError(f"checkpoint at {target} is not an EcosystemState")
        return state

    def read_manifest(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path is not None else self.latest_path()
        if target is None:
            raise FileNotFoundError(f"no checkpoint under {self.state_dir}")
        return json.loads((Path(target) / MANIFEST).read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=True), encoding="utf-8")

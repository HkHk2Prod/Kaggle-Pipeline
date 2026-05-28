"""A lightweight append-only experiment recorder (JSON Lines).

Persists feature genomes, model genomes and mutation records so a run is auditable
and reproducible from disk. Intentionally minimal -- one JSONL file per record
type. A future store could index by hash, support resume/warm-start, or back onto
a database; the controller only depends on the small ``record_*`` surface here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_pipeline.evolution.storage.serialization import to_json


class ExperimentStore:
    """Appends serialized records to per-type JSONL files under a directory."""

    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _append(self, filename: str, obj: Any) -> None:
        if self.directory is None:
            return
        line = to_json(obj)
        with (self.directory / filename).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def record_feature(self, genome: Any) -> None:
        self._append("features.jsonl", genome)

    def record_model(self, genome: Any) -> None:
        self._append("models.jsonl", genome)

    def record_mutation(self, record: Any) -> None:
        self._append("mutations.jsonl", record)

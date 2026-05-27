"""JSON serialization helpers for evolutionary records.

Every major object (recipes, genomes, genes, score sets, mutation records,
results) exposes ``to_serializable()`` returning JSON-compatible structures. These
helpers turn those into JSON text/files without each call site re-deriving the
conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def serialize(obj: Any) -> Any:
    """Return a JSON-compatible view of ``obj`` (uses ``to_serializable`` if present)."""
    if hasattr(obj, "to_serializable"):
        return obj.to_serializable()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj


def to_json(obj: Any, *, indent: int | None = None) -> str:
    """Serialize ``obj`` to a JSON string."""
    return json.dumps(serialize(obj), indent=indent, default=str, ensure_ascii=True)


def dump(obj: Any, path: str | Path, *, indent: int | None = 2) -> None:
    """Write ``obj`` as JSON to ``path``."""
    Path(path).write_text(to_json(obj, indent=indent), encoding="utf-8")

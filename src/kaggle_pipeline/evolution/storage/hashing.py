"""Canonical, stable hashing of recipes and genomes.

Reproducibility rests on hashing the *canonical* structure of an object, never
its human-readable name. The same recipe/genome must always produce the same
hash across processes and Python runs, so:

* dicts are serialised with sorted keys;
* floats are normalised to a fixed precision (and ``-0.0`` collapses to ``0.0``);
* NaN/Inf get explicit sentinels (plain ``json`` cannot round-trip them);
* dataclasses are converted to dicts recursively.

Callers (recipes, genomes) are responsible for *canonicalising their own
content* first -- e.g. sorting parent IDs for commutative operators -- before
handing it here. This module only guarantees a deterministic digest of whatever
structure it is given.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

# Length of the short hash suffix used in human-readable feature/model names.
SHORT_HASH_LENGTH = 6
# Significant figures kept when normalising floats for hashing.
_FLOAT_PRECISION = 12


def _canonical(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-stable structure."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(asdict(obj))
    if isinstance(obj, dict):
        # Sort by the string form of the key for a deterministic order.
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool):
        # Must precede the int/float branch: bool is a subclass of int.
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__+inf__" if obj > 0 else "__-inf__"
        # Normalise representation and collapse signed zero.
        return float(f"{obj:.{_FLOAT_PRECISION}g}") + 0.0
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    # Fall back to the repr for anything exotic so hashing never crashes; callers
    # should avoid putting non-serialisable objects into recipes/genomes.
    return repr(obj)


def canonical_json(obj: Any) -> str:
    """Return a canonical JSON string for ``obj`` (sorted keys, normalised floats)."""
    return json.dumps(
        _canonical(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_hash(obj: Any) -> str:
    """Return the full hex SHA-256 digest of ``obj``'s canonical form."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, length: int = SHORT_HASH_LENGTH) -> str:
    """Return a short hex digest of ``obj`` for use in human-readable names.

    Accepts either an arbitrary object (hashed via :func:`stable_hash`) or an
    already-computed 64-char hex digest (sliced directly).
    """
    if isinstance(obj, str) and len(obj) == 64 and all(c in "0123456789abcdef" for c in obj):
        full = obj
    else:
        full = stable_hash(obj)
    return full[:length]

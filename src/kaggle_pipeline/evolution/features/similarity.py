"""Sparse top-k similarity between global feature materializations.

Because features are global, redundancy is computed between *global* feature
value vectors on a fixed reference sample -- never inside private model genes.
We keep only the top-k strongest neighbours per feature (not a forever-dense
matrix) and the standardized sample vector needed to compare against newcomers.

Numeric similarity (``|Pearson|`` on the sample) ships first; categorical
similarity is planned and tracked via value-hash duplicate detection meanwhile.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _standardize(values: np.ndarray) -> np.ndarray | None:
    """Return a zero-mean unit-std vector (NaNs -> 0), or ``None`` if constant."""
    x = np.asarray(values, dtype=float).ravel()
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < _EPS:
        return None
    z = (x - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


class FeatureSimilarity:
    """Maintains standardized sample vectors and top-k absolute-correlation neighbours."""

    def __init__(self, *, top_k: int = 5):
        self.top_k = top_k
        self._vectors: dict[str, np.ndarray] = {}
        self._neighbors: dict[str, list[tuple[str, float]]] = {}
        self._value_hashes: dict[str, str] = {}

    def has_vector(self, feature_id: str) -> bool:
        return feature_id in self._vectors

    def set_value_hash(self, feature_id: str, value_hash: str) -> str | None:
        """Record a feature's value hash; return an existing feature with the same one."""
        for other, h in self._value_hashes.items():
            if h == value_hash and other != feature_id:
                return other
        self._value_hashes[feature_id] = value_hash
        return None

    def update_for_feature(self, feature_id: str, values: np.ndarray) -> float:
        """Add/update ``feature_id``'s vector and return its redundancy (max |corr|).

        Compares against every other stored numeric vector, records the top-k
        strongest neighbours for both sides, and returns the maximum absolute
        correlation -- the feature's redundancy score (0 if it has no peers or is
        constant on the sample).
        """
        vec = _standardize(values)
        if vec is None:
            # Constant on the sample: no usable numeric similarity.
            self._neighbors[feature_id] = []
            return 0.0

        n = vec.size
        sims: list[tuple[str, float]] = []
        for other_id, other_vec in self._vectors.items():
            if other_id == feature_id or other_vec.size != n:
                continue
            corr = abs(float(np.dot(vec, other_vec) / n))
            if not np.isfinite(corr):
                continue
            sims.append((other_id, corr))
            self._record_neighbor(other_id, feature_id, corr)

        self._vectors[feature_id] = vec
        sims.sort(key=lambda t: t[1], reverse=True)
        self._neighbors[feature_id] = sims[: self.top_k]
        return sims[0][1] if sims else 0.0

    def _record_neighbor(self, owner: str, other: str, corr: float) -> None:
        bucket = self._neighbors.setdefault(owner, [])
        bucket.append((other, corr))
        bucket.sort(key=lambda t: t[1], reverse=True)
        del bucket[self.top_k :]

    def redundancy_of_vector(self, values: np.ndarray) -> float:
        """Max ``|corr|`` of a transient vector against stored vectors (no insert).

        Used to score a generation candidate before it is admitted, so a redundant
        feature can be rejected without polluting the stored similarity state.
        """
        vec = _standardize(values)
        if vec is None or not self._vectors:
            return 0.0
        best = 0.0
        for other_vec in self._vectors.values():
            if other_vec.size != vec.size:
                continue
            corr = abs(float(np.dot(vec, other_vec) / vec.size))
            if np.isfinite(corr) and corr > best:
                best = corr
        return best

    def get(self, a: str, b: str) -> float:
        """Absolute correlation between two features if both have sample vectors."""
        va, vb = self._vectors.get(a), self._vectors.get(b)
        if va is None or vb is None or va.size != vb.size:
            return 0.0
        return abs(float(np.dot(va, vb) / va.size))

    def redundancy(self, feature_id: str) -> float:
        neighbors = self._neighbors.get(feature_id)
        return neighbors[0][1] if neighbors else 0.0

    def neighbors(self, feature_id: str) -> list[tuple[str, float]]:
        return list(self._neighbors.get(feature_id, []))

    def is_duplicate(self, feature_id: str, *, threshold: float = 0.999) -> str | None:
        """Return a near-duplicate feature id (corr >= threshold), else ``None``."""
        for other, corr in self._neighbors.get(feature_id, []):
            if corr >= threshold:
                return other
        return None

    def remove(self, feature_id: str) -> None:
        self._vectors.pop(feature_id, None)
        self._neighbors.pop(feature_id, None)
        self._value_hashes.pop(feature_id, None)
        for bucket in self._neighbors.values():
            bucket[:] = [(o, c) for (o, c) in bucket if o != feature_id]

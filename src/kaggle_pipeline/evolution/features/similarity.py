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

from kaggle_pipeline.evolution.utils.arrays import abs_correlation, standardize_for_correlation


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
        vec = standardize_for_correlation(values)
        if vec is None:
            # Constant on the sample: no usable numeric similarity.
            self._neighbors[feature_id] = []
            return 0.0

        sims: list[tuple[str, float]] = []
        for other_id, other_vec in self._vectors.items():
            if other_id == feature_id:
                continue
            corr = abs_correlation(vec, other_vec)
            if corr is None:
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
        vec = standardize_for_correlation(values)
        if vec is None or not self._vectors:
            return 0.0
        best = 0.0
        for other_vec in self._vectors.values():
            corr = abs_correlation(vec, other_vec)
            if corr is not None and corr > best:
                best = corr
        return best

    def get(self, a: str, b: str) -> float:
        """Absolute correlation between two features if both have sample vectors."""
        va, vb = self._vectors.get(a), self._vectors.get(b)
        if va is None or vb is None:
            return 0.0
        corr = abs_correlation(va, vb)
        return corr if corr is not None else 0.0

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

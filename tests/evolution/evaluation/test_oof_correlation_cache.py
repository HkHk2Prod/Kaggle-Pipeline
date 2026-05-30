"""Incremental cache of residual-error correlations for OOF predictions."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.evaluation.oof_correlation_cache import OOFCorrelationCache
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore


def _binary_y(seed: int = 0, n: int = 200) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(size=n) > 0.5).astype(int)


def test_add_caches_self_vector_and_all_pairs():
    cache = OOFCorrelationCache()
    y = _binary_y(0)
    rng = np.random.default_rng(1)

    cache.set_target(y, {})
    cache.add("a", rng.uniform(size=200))
    cache.add("b", rng.uniform(size=200))
    cache.add("c", rng.uniform(size=200))

    assert cache.size() == 3
    assert cache.has("a") and cache.has("b") and cache.has("c")
    # Three models -> three unordered pairs.
    assert cache.correlation("a", "b") is not None
    assert cache.correlation("b", "c") is not None
    assert cache.correlation("a", "c") is not None


def test_correlation_is_symmetric_and_self_is_none():
    cache = OOFCorrelationCache()
    y = _binary_y(2)
    cache.set_target(y, {})
    cache.add("m", np.linspace(0, 1, 200))
    cache.add("n", np.linspace(0, 1, 200) ** 2)

    assert cache.correlation("m", "n") == cache.correlation("n", "m")
    assert cache.correlation("m", "m") is None  # no self-pair entry


def test_remove_drops_vector_and_every_pair_with_it():
    cache = OOFCorrelationCache()
    y = _binary_y(3)
    rng = np.random.default_rng(4)
    cache.set_target(y, {})
    for mid in ("a", "b", "c"):
        cache.add(mid, rng.uniform(size=200))

    cache.remove("b")
    assert not cache.has("b")
    assert cache.correlation("a", "b") is None
    assert cache.correlation("b", "c") is None
    # The pair that did not involve "b" survives.
    assert cache.correlation("a", "c") is not None


def test_replace_drops_stale_pairs_before_recomputing():
    cache = OOFCorrelationCache()
    y = _binary_y(5)
    rng = np.random.default_rng(6)
    cache.set_target(y, {})
    cache.add("a", rng.uniform(size=200))
    cache.add("b", rng.uniform(size=200))
    first = cache.correlation("a", "b")

    # Replace b's OOF with one that anti-correlates with a's residuals.
    a_oof = next(iter(cache._z.values()))  # standardized residual is fine for the test
    cache.add("b", -a_oof)
    replaced = cache.correlation("a", "b")
    assert replaced is not None
    assert replaced != first  # stale pair was not reused


def test_set_target_rebuilds_from_existing_oofs():
    # When y first becomes known, previously stored OOFs should be backfilled.
    store_like: dict[str, np.ndarray] = {
        "a": np.linspace(0, 1, 200),
        "b": np.linspace(0, 1, 200) ** 1.05,
    }
    cache = OOFCorrelationCache()
    cache.set_target(_binary_y(7), store_like)
    assert cache.size() == 2
    assert cache.correlation("a", "b") is not None


def test_set_target_same_array_is_noop():
    cache = OOFCorrelationCache()
    y = _binary_y(8)
    cache.set_target(y, {"a": np.linspace(0, 1, 200)})
    snapshot = cache.correlation("a", "a")  # None either way
    # Same array reference: no rebuild, internal vectors intact.
    z_before = cache._z["a"]
    cache.set_target(y, {})  # would normally wipe, but identity check skips it
    assert cache._z["a"] is z_before
    assert cache.correlation("a", "a") == snapshot


def test_add_is_noop_without_target():
    cache = OOFCorrelationCache()
    cache.add("a", np.linspace(0, 1, 200))
    assert cache.size() == 0


def test_oof_store_round_trip_keeps_cache_consistent():
    # Storing through the OOFStore (the actual integration path) should
    # populate the cache after ``set_residual_target`` has been called.
    store = OOFStore()
    y = _binary_y(9)
    rng = np.random.default_rng(10)
    store.store("a", rng.uniform(size=200))  # no target yet
    assert store.correlation_cache.size() == 0  # cache stays empty

    store.set_residual_target(y)  # backfills "a"
    assert store.correlation_cache.has("a")

    store.store("b", rng.uniform(size=200))  # target bound, "b" indexed live
    assert store.correlation_cache.correlation("a", "b") is not None

    store.remove("a")
    assert not store.correlation_cache.has("a")
    assert store.correlation_cache.correlation("a", "b") is None


def test_oof_store_does_not_pickle_correlation_cache():
    # The cache is a derived view; rebuilding on resume is cheaper than
    # shipping it inside the checkpoint. ``__getstate__`` drops it.
    import pickle

    store = OOFStore()
    store.set_residual_target(_binary_y(11))
    store.store("a", np.linspace(0, 1, 200))
    assert store.correlation_cache.has("a")

    restored = pickle.loads(pickle.dumps(store))
    assert restored.correlation_cache.size() == 0
    # OOFs survive; cache repopulates once the target is rebound.
    restored.set_residual_target(_binary_y(11))
    assert restored.correlation_cache.has("a")

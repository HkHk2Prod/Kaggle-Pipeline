"""Canonical hashing is stable, order-sensitive on dicts, and float-robust."""

from __future__ import annotations

import math

from kaggle_pipeline.evolution.storage.hashing import (
    canonical_json,
    short_hash,
    stable_hash,
)


def test_stable_hash_is_deterministic():
    obj = {"b": 1, "a": [3, 2, 1], "c": {"y": 2, "x": 1}}
    assert stable_hash(obj) == stable_hash(obj)


def test_dict_key_order_does_not_change_hash():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_list_order_does_change_hash():
    assert stable_hash([1, 2, 3]) != stable_hash([3, 2, 1])


def test_float_normalization_collapses_signed_zero():
    assert canonical_json({"x": -0.0}) == canonical_json({"x": 0.0})


def test_nan_and_inf_are_hashable():
    # Plain json cannot round-trip these; our canonicaliser uses sentinels.
    h_nan = stable_hash({"x": math.nan})
    h_inf = stable_hash({"x": math.inf})
    assert h_nan != h_inf
    assert stable_hash({"x": math.nan}) == h_nan


def test_short_hash_slices_existing_digest():
    full = stable_hash({"a": 1})
    assert short_hash(full, 6) == full[:6]
    assert len(short_hash({"a": 1}, 8)) == 8

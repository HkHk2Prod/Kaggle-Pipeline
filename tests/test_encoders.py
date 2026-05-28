"""Tests for :class:`FrequencyEncoder`."""

from __future__ import annotations

import pandas as pd

from kaggle_pipeline.preprocessing import FrequencyEncoder

N_DRIVERS = 40


def test_frequency_encoder_maps_to_training_frequencies():
    enc = FrequencyEncoder().fit(pd.DataFrame({"c": ["a", "a", "a", "b"]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "z"]}))  # "z" unseen
    assert list(out["c"]) == [0.75, 0.25, 0.0]  # unseen level -> 0.0
    assert out["c"].dtype == float


def test_frequency_encoder_is_one_column_in_one_column_out():
    df = pd.DataFrame({"driver": [f"d{i % N_DRIVERS}" for i in range(200)]})
    out = FrequencyEncoder().fit_transform(df)
    assert out.shape == (200, 1)  # never widens, unlike one-hot

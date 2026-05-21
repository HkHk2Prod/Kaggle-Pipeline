"""Tests for the leaderboard capacity logic and persistence."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.search.leaderboard import LeaderBoard, ModelClass, ModelEntry


def _entry(score: float, tmp_path) -> ModelEntry:
    # Each entry points at a real (empty) file so delete_file has something to remove.
    path = tmp_path / f"m_{score}"
    path.write_text("")
    return ModelEntry(score=score, name=path.name, file_path=str(path), compute_time=1)


def test_model_class_keeps_top_entries_sorted_desc(tmp_path):
    cl = ModelClass(lower=1, upper=2)
    cl.insert(_entry(0.5, tmp_path))
    cl.insert(_entry(0.9, tmp_path))
    cl.insert(_entry(0.7, tmp_path))  # full -> evicts the worst (0.5)

    scores = [e.score for e in cl.entries]
    assert scores == [0.9, 0.7]
    assert not (tmp_path / "m_0.5").exists()  # evicted file deleted


def test_model_class_rejects_worse_when_full(tmp_path):
    cl = ModelClass(lower=1, upper=1)
    cl.insert(_entry(0.8, tmp_path))
    cl.insert(_entry(0.3, tmp_path))  # worse than current worst -> rejected
    assert [e.score for e in cl.entries] == [0.8]
    assert not (tmp_path / "m_0.3").exists()


def test_leaderboard_save_load_preserves_runtime_state(tmp_path):
    seed_a = np.random.SeedSequence(1)
    board = LeaderBoard(num_models=10, storage_dir=tmp_path, seed_seq=seed_a)
    board.add_class("LogisticRegression", lower=1, upper=3)
    board.classes["LogisticRegression"].insert(_entry(0.6, tmp_path))
    board.save()

    seed_b = np.random.SeedSequence(2)
    fresh = LeaderBoard(num_models=10, storage_dir=tmp_path, seed_seq=seed_b)
    assert fresh.load() is True
    # Classes/entries restored from disk...
    assert "LogisticRegression" in fresh.classes
    assert len(fresh.classes["LogisticRegression"]) == 1
    # ...but the *current* seed sequence is kept, not the pickled one.
    assert fresh.seed_seq is seed_b

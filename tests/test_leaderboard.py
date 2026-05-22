"""Tests for the leaderboard capacity logic and persistence."""

from __future__ import annotations

from pathlib import Path

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


def test_add_class_resolves_int_and_float_bounds(tmp_path):
    # int bounds are absolute; float bounds are fractions of num_models.
    board = LeaderBoard(num_models=300, storage_dir=tmp_path, seed_seq=np.random.SeedSequence(0))
    board.add_class("frac", lower=0.05, upper=0.30)  # 5% / 30% of 300
    board.add_class("abs", lower=3, upper=10)  # taken literally
    assert (board.classes["frac"].lower, board.classes["frac"].upper) == (15, 90)
    assert (board.classes["abs"].lower, board.classes["abs"].upper) == (3, 10)


def test_add_class_rounds_fractions_up(tmp_path):
    board = LeaderBoard(num_models=100, storage_dir=tmp_path, seed_seq=np.random.SeedSequence(0))
    board.add_class("rf", lower=0.025, upper=0.105)  # 2.5 -> 3, 10.5 -> 11
    assert (board.classes["rf"].lower, board.classes["rf"].upper) == (3, 11)


def _saturated_board(seed_seq, storage_dir):
    """Two classes, each above its lower bound, so ``get`` takes the softmax path.

    Class "A" scores well above "B"; a correct softmax should favour "A".
    """
    board = LeaderBoard(num_models=100, storage_dir=storage_dir, seed_seq=seed_seq)
    board.add_class("A", lower=1, upper=5)
    board.add_class("B", lower=1, upper=5)
    for score in (0.90, 0.80):
        board.classes["A"].insert(_entry(score, storage_dir))
    for score in (0.50, 0.40):
        board.classes["B"].insert(_entry(score, storage_dir))
    return board


def test_get_is_reproducible_under_same_seed(tmp_path):
    # Two boards seeded identically must produce the same selection sequence:
    # the choice draws from the seed sequence, not the global RNG.
    board1 = _saturated_board(np.random.SeedSequence(123), tmp_path / "a")
    board2 = _saturated_board(np.random.SeedSequence(123), tmp_path / "b")
    seq1 = [board1.get() for _ in range(25)]
    seq2 = [board2.get() for _ in range(25)]
    assert seq1 == seq2


def test_get_softmax_favours_the_higher_scoring_class(tmp_path):
    board = _saturated_board(np.random.SeedSequence(7), tmp_path)
    picks = [board.get() for _ in range(300)]
    assert picks.count("A") > picks.count("B")  # higher mean score -> picked more


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


def test_load_rebases_entry_paths_to_current_storage_dir(tmp_path):
    """A board saved elsewhere reloads with entry paths pointing at the new dir.

    Mirrors a warm-start: a previous run saved the board (and model files) under
    one dir; on resume the files are copied beside the board in *this* run's
    storage dir, so loaded entries must point there, not at the stale path.
    """
    old = tmp_path / "old_models"
    old.mkdir()
    board = LeaderBoard(num_models=10, storage_dir=old, seed_seq=np.random.SeedSequence(1))
    board.add_class("LogisticRegression", lower=1, upper=3)
    board.classes["LogisticRegression"].insert(_entry(0.6, old))
    board.save()

    # Resume from a *different* storage dir holding a copy of the board file.
    new = tmp_path / "new_models"
    new.mkdir()
    (new / "LeaderBoard").write_bytes((old / "LeaderBoard").read_bytes())
    fresh = LeaderBoard(num_models=10, storage_dir=new, seed_seq=np.random.SeedSequence(2))
    assert fresh.load() is True

    entry = fresh.classes["LogisticRegression"].entries[0]
    assert Path(entry.file_path).parent == new  # rebased onto the current dir
    assert Path(entry.file_path).name == entry.name

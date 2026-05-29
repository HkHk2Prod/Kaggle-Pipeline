"""Submission CSV construction and logging for :class:`KagglePipeline`.

The pipeline owns the model lifecycle; this module owns the test-time decoding
and the writing of a competition-ready CSV. It is split out so the pipeline
itself does not also have to know about per-class column layouts, id alignment
between test and sample-submission frames, or the verbosity-tiered summary lines
that get printed after a write.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle_pipeline.evolution.runtime import RuntimeManager


@dataclass(frozen=True)
class SubmissionWriter:
    """Decode pipeline predictions into a submission DataFrame.

    The writer is constructed from the immutable problem state captured at
    ``fit`` time (task, class labels, prediction aim, id metadata) so each
    instance is safe to reuse across calls and can be tested without spinning
    up a full pipeline.
    """

    task: str
    classes: np.ndarray | None
    prediction_aim: str
    id_col: str
    test_ids: Any
    test_has_ids: bool

    def build_frame(
        self,
        predictions: np.ndarray,
        sample: pd.DataFrame | None,
        *,
        target_col: str = "target",
    ) -> pd.DataFrame:
        """Return the DataFrame to write as the submission CSV."""
        if sample is not None:
            return self._from_sample(sample, predictions)
        return self._fallback(predictions, target_col)

    def decode_single(self, proba: np.ndarray) -> np.ndarray:
        """Decode predictions into one submission column per ``prediction_aim``."""
        if self.task != "classification" or self.classes is None:
            return proba.ravel()
        if self.prediction_aim == "category":
            return self.classes[proba.argmax(axis=1)] if proba.ndim == 2 else proba.ravel()
        # probability
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]  # positive class
        if proba.ndim == 2:
            return self.classes[proba.argmax(axis=1)]  # multiclass single-col: best-effort label
        return proba.ravel()

    def _from_sample(self, sample: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """Build a submission matching the sample_submission's columns/structure.

        ``predictions`` are in test-row order (the order of ``self.test_ids``).
        The sample submission is *not* guaranteed to share that order, so rows are
        matched to the sample on the id column rather than by position -- aligning
        by position silently scrambles every prediction whenever the two files are
        sorted differently. The id sets are asserted equal first so a mismatch
        fails loudly instead of yielding a submission full of nulls, and the
        sample's column order is preserved. When the test set carried no id column
        there is nothing to join on, so we fall back to positional alignment.
        """
        columns = list(sample.columns)
        id_col = self.id_col if self.id_col in columns else columns[0]
        target_cols = [c for c in columns if c != id_col]
        proba = np.asarray(predictions, dtype=float)

        preds = pd.DataFrame({id_col: np.asarray(self.test_ids)})
        if self.test_has_ids:
            preds[id_col] = preds[id_col].astype(sample[id_col].dtype)
        if len(target_cols) == 1:
            preds[target_cols[0]] = self.decode_single(proba)
        else:
            # One probability column per class (sample order assumed = class order).
            matrix = proba if proba.ndim == 2 else proba.reshape(-1, 1)
            for i, col in enumerate(target_cols):
                preds[col] = matrix[:, i] if i < matrix.shape[1] else 0.0

        if not self.test_has_ids:
            return preds[columns]  # no real ids to align on; keep test order

        sample_ids = set(sample[id_col].tolist())
        pred_ids = set(preds[id_col].tolist())
        if sample_ids != pred_ids:
            raise ValueError(
                f"Test ids and sample-submission ids do not match on {id_col!r}: "
                f"{len(pred_ids)} test id(s) vs {len(sample_ids)} sample id(s), "
                f"{len(pred_ids & sample_ids)} in common. Cannot build a submission."
            )
        result = sample.drop(columns=target_cols).merge(preds, on=id_col, how="left")
        if len(result) != len(sample):
            raise ValueError(
                f"Joining predictions on {id_col!r} changed the row count "
                f"({len(sample)} -> {len(result)}); the id column is not unique."
            )
        return result[columns]  # preserve the sample's column order

    def _fallback(self, predictions: np.ndarray, target_col: str) -> pd.DataFrame:
        decoded = self.decode_single(np.asarray(predictions, dtype=float))
        return pd.DataFrame({self.id_col: self.test_ids, target_col: np.asarray(decoded).ravel()})


def submission_summary_lines(
    path: Path,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    ensemble_result: Any,
    population_lookup: Callable[[str], Any] | None = None,
) -> tuple[list[tuple[str, int]], str | None]:
    """Produce the (line, verbosity_level) tuples for the post-write summary.

    Returns a ``(summary_lines, composition_block)`` pair. ``summary_lines``
    are one-liners (each tagged with the verbosity level at which it should
    print); ``composition_block`` is the multi-line per-member breakdown for
    NORMAL+ users, or ``None`` when there is nothing to show. Splitting it
    that way lets the caller still own ``self.log`` semantics without baking
    its verbosity enum into this module.

    ``population_lookup`` resolves an ensemble member id to its genome (with
    ``family`` and ``score_set`` attributes); pass ``None`` to skip composition.
    """
    from kaggle_pipeline.evolution.logging_utils import Verbosity

    proba = np.asarray(predictions, dtype=float)
    finite = bool(np.isfinite(proba).all())
    lines: list[tuple[str, int]] = []
    if ensemble_result is None:
        lines.append(
            (
                f"submission summary: file={path} rows={len(frame)} cols={list(frame.columns)} "
                f"finite_predictions={finite} -- ensemble=disabled (best single model)",
                Verbosity.SUMMARY,
            )
        )
        return lines, None

    score = f"{ensemble_result.oof_score:.4f}" if ensemble_result.oof_score is not None else "n/a"
    lines.append(
        (
            f"submission summary: file={path} rows={len(frame)} "
            f"cols={list(frame.columns)} finite_predictions={finite}",
            Verbosity.SUMMARY,
        )
    )
    note = f" note={ensemble_result.note}" if ensemble_result.note else ""
    lines.append(
        (
            f"ensemble: strategy={ensemble_result.status} members={ensemble_result.n_members} "
            f"oof_score={score}{note}",
            Verbosity.SUMMARY,
        )
    )
    if population_lookup is None or not ensemble_result.member_ids:
        return lines, None
    rows: list[str] = []
    for mid in ensemble_result.member_ids:
        genome = population_lookup(mid)
        if genome is None:
            continue
        weight = ensemble_result.weights.get(mid, 0.0)
        score_set = genome.score_set
        member_score = f"{score_set.score:.4f}" if score_set else "n/a"
        member_std = f"±{score_set.score_std:.4f}" if score_set else ""
        member_time = f" time={score_set.compute_time:.2f}s" if score_set else ""
        rows.append(
            f"  - {mid} [{genome.family}] weight={weight:.3f} "
            f"score={member_score}{member_std}{member_time}"
        )
    composition = ("ensemble composition:\n" + "\n".join(rows)) if rows else None
    return lines, composition


def submission_skip_reason(
    *,
    make_submission_on_run: bool,
    has_test_features: bool,
    runtime: RuntimeManager | None,
) -> str | None:
    """Decide whether the post-run auto-submission should be skipped.

    Returns ``None`` when submission should proceed; otherwise a SUMMARY-level
    message ready to be logged by the caller. The branches mirror the three
    guards the pipeline needs: feature off, no test data given to ``fit``,
    and not enough budget left within the reserved submission window.
    """
    if not make_submission_on_run:
        return ""  # silent skip: the flag is simply off
    if not has_test_features:
        return "make_submission_on_run set but no test data was given to fit(); skipping"
    if runtime is not None and not runtime.has_time_for_submission():
        return (
            f"not enough time for submission within the reserved window; skipping "
            f"(remaining={runtime.remaining_submission_seconds():.0f}s, "
            f"estimate={runtime.submission_time_reserve_seconds:.0f}s)"
        )
    return None

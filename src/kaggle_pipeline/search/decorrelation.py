"""De-correlate the leaderboard: drop models that make the same mistakes.

When one model class dominates the search, the leaderboard fills with near-copies
of the same estimator. Their out-of-fold predictions are almost identical, so the
stacking meta-model has nothing to combine and the ensemble can't beat the single
best model. This module decides which models are *redundant* so the search can
delete them.

Redundancy is judged on the **residuals** ``y - y_oof`` rather than the raw OOF
predictions, so two models count as redundant only when their *errors* line up --
models that are confidently wrong in different places are kept, because that is
exactly the diversity an ensemble needs.

The test is deliberately conservative on small data: rather than compare the
observed residual correlation to a fixed cut-off, we take the one-sided lower
confidence bound of that correlation (Fisher's z-transform) and compare *that* to
``tau``. The confidence interval widens as the dataset shrinks, so a high
correlation measured on few rows -- which may be noise -- does not trigger a
deletion.

Possible improvement: today this runs as a per-batch pass over the whole board
(``Judge.prune_correlated_models``), which re-compares every pair each batch even
though the board is already de-correlated. Folding the check into
``LeaderBoard.add`` -- reject a newcomer that is redundant with a better kept
model *before* it is admitted -- would be cheaper (only newcomers vs the kept set)
and would stop a redundant model from ever evicting a diverse one to make room. It
was kept as a batch pass for simplicity and so it also cleans warm-started boards;
see the project notes for the trade-off.
"""

from __future__ import annotations

import numpy as np

# One-sided 95% normal quantile. Fixed (not a config knob) to keep the rule
# simple: we prune only when we are ~95% confident the residual correlation
# really exceeds ``tau``.
Z_ONE_SIDED_95 = 1.645


def residual_correlation_lower_bound(r: float, n_eff: int, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided lower confidence bound on a correlation ``r`` from ``n_eff`` points.

    We don't compare the *observed* residual correlation to ``tau`` directly: on
    little data a high correlation can be sampling noise. Fisher's z-transform
    ``atanh(r)`` is approximately normal with standard error ``1/sqrt(n_eff - 3)``,
    so we shift down by ``z`` standard errors (the 95% one-sided quantile) and map
    back with ``tanh``. The width of this interval -- and therefore how much
    evidence we demand before calling two models redundant -- scales with the
    dataset size: ``n_eff`` is the number of training rows, so smaller datasets
    require a *higher* observed correlation before anything is pruned. For a large
    dataset the bound collapses onto ``r`` and the rule becomes "prune above tau".
    """
    # With n_eff <= 3 the standard error is undefined (sqrt of <= 0); we simply
    # never have enough data to be confident, so report a bound below any tau.
    if n_eff <= 3:
        return -1.0
    # Clip so atanh stays finite when two models are (near-)identical (r == 1).
    r = float(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12))
    se = 1.0 / np.sqrt(n_eff - 3)
    return float(np.tanh(np.arctanh(r) - z * se))


def standardize(residual: np.ndarray) -> np.ndarray | None:
    """Centre and unit-normalise a residual vector so a dot product is Pearson r.

    Returns ``None`` for a zero-variance residual (e.g. a model that is perfect on
    the OOF rows): no correlation is defined, so such a model is neither pruned nor
    used as a yardstick for others. Stored as float32 to bound memory when many
    long residual vectors are cached at once; the precision is ample for a ~0.98
    gate. The result is reused across batches (residuals don't change once a model
    is on the board), so each model is standardised exactly once.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    r -= r.mean()
    norm = np.linalg.norm(r)
    if norm == 0.0:
        return None
    return (r / norm).astype(np.float32)


def select_redundant(
    units: list[np.ndarray | None],
    *,
    n_eff: int,
    tau: float,
    z: float = Z_ONE_SIDED_95,
) -> set[int]:
    """Greedy, score-ordered de-correlation over standardised residuals.

    ``units`` are unit residual vectors (from :func:`standardize`, ``None`` for a
    degenerate residual) ordered best-score first. Walking from the best model
    down, a model is marked redundant when the lower confidence bound (see
    :func:`residual_correlation_lower_bound`) on the correlation of its residual
    with an already-kept, higher-scoring model's residual exceeds ``tau`` -- i.e.
    we are confident the two make the same mistakes. Returns the indices to drop.

    We keep the better model and drop the worse. A plausibly better strategy would
    be to *average* the redundant pair (or keep the average only when it beats the
    best of the two): two correlated-but-not-identical models still carry some
    independent noise, so averaging them would cut variance rather than discard
    information. Left as a future improvement.
    """
    kept_units: list[np.ndarray] = []
    dropped: set[int] = set()
    for i, unit in enumerate(units):
        if unit is None:
            continue
        is_redundant = any(
            residual_correlation_lower_bound(float(unit @ kept), n_eff, z) > tau
            for kept in kept_units
        )
        if is_redundant:
            dropped.add(i)
        else:
            kept_units.append(unit)
    return dropped


def select_redundant_indices(
    residuals: list[np.ndarray],
    *,
    n_eff: int,
    tau: float,
    z: float = Z_ONE_SIDED_95,
) -> set[int]:
    """Convenience wrapper of :func:`select_redundant` that standardises first."""
    return select_redundant(
        [standardize(r) for r in residuals], n_eff=n_eff, tau=tau, z=z
    )

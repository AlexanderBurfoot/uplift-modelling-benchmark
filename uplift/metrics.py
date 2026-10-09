"""Evaluation for uplift models.

Ranking metrics (Qini) work on real data: they never reference an
individual effect, only the incremental conversions from targeting the top
k by score versus k at random.

Oracle metrics (PEHE, effect correlation) need the true per-individual
effect, so synthetic data only. Both are reported because a model can rank
well while estimating badly.

Several incompatible Qini definitions circulate. This one is

    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

over the top k by score, where the scaling term puts the control arm on the
treated arm's footing. ``qini_auc`` is the area between that curve and the
line a random ranking traces.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "qini_curve",
    "qini_auc",
    "qini_coefficient",
    "incremental_at_k",
    "pehe",
    "effect_correlation",
]


def _prepare(y, treatment, score):
    y = np.asarray(y, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if not (len(y) == len(treatment) == len(score)):
        raise ValueError("y, treatment and score must have the same length")
    if len(y) == 0:
        raise ValueError("empty input")
    return y, treatment, score


def qini_curve(y, treatment, score) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental conversions against targeting depth.

    Returns ``(fraction_targeted, incremental_conversions)``, length n + 1,
    both starting at zero. Ties keep input order via a stable sort.
    """
    y, treatment, score = _prepare(y, treatment, score)
    order = np.argsort(-score, kind="stable")
    y_o, t_o = y[order], treatment[order]

    cum_y_t = np.cumsum(y_o * t_o)
    cum_y_c = np.cumsum(y_o * (1.0 - t_o))
    cum_n_t = np.cumsum(t_o)
    cum_n_c = np.cumsum(1.0 - t_o)

    safe_n_c = np.maximum(cum_n_c, 1)
    scaled_control = np.where(cum_n_c > 0, cum_y_c * cum_n_t / safe_n_c, 0.0)
    gains = cum_y_t - scaled_control

    n = len(y)
    fraction = np.arange(0, n + 1) / n
    return fraction, np.concatenate([[0.0], gains])


def qini_auc(y, treatment, score) -> float:
    """Area between the Qini curve and the random-targeting diagonal.

    Units are incremental conversions. Negative means the ranking is worse
    than random, which is the usual result for a response model.
    """
    fraction, gains = qini_curve(y, treatment, score)
    diagonal = fraction * gains[-1]
    return float(np.trapezoid(gains - diagonal, fraction))


def qini_coefficient(y, treatment, score, reference_score) -> float:
    """``qini_auc`` as a fraction of a reference ranking's area.

    The reference is normally the oracle ranking by true effect. No agreed
    construction exists for an optimal curve from observed outcomes alone,
    so the reference is required rather than guessed.

    Values above 1.0 indicate noise in the denominator, not a model beating
    the oracle.
    """
    denominator = qini_auc(y, treatment, reference_score)
    if denominator == 0:
        raise ValueError("reference ranking has zero Qini area; cannot normalise")
    return qini_auc(y, treatment, score) / denominator


def incremental_at_k(y, treatment, score, k: float) -> float:
    """Incremental conversions from targeting the top ``k`` fraction."""
    if not 0.0 < k <= 1.0:
        raise ValueError("k must lie in (0, 1]")
    fraction, gains = qini_curve(y, treatment, score)
    index = int(round(k * (len(gains) - 1)))
    return float(gains[index])


def pehe(true_tau, predicted_tau) -> float:
    """Root mean squared error on the individual treatment effect.

    Precision in estimating heterogeneous effects. Oracle metric.
    """
    true_tau = np.asarray(true_tau, dtype=float).ravel()
    predicted_tau = np.asarray(predicted_tau, dtype=float).ravel()
    if len(true_tau) != len(predicted_tau):
        raise ValueError("inputs must have the same length")
    return float(np.sqrt(np.mean((true_tau - predicted_tau) ** 2)))


def effect_correlation(true_tau, predicted_tau, method: str = "spearman") -> float:
    """Correlation between true and predicted individual effects.

    Spearman by default: targeting depends on ordering, not calibration.
    """
    true_tau = np.asarray(true_tau, dtype=float).ravel()
    predicted_tau = np.asarray(predicted_tau, dtype=float).ravel()
    if method == "spearman":
        from scipy.stats import spearmanr

        return float(spearmanr(true_tau, predicted_tau).statistic)
    if method == "pearson":
        return float(np.corrcoef(true_tau, predicted_tau)[0, 1])
    raise ValueError("method must be 'spearman' or 'pearson'")

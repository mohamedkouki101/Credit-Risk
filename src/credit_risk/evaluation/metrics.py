"""Discrimination, calibration and comparison metrics (EV-01).

Conventions used everywhere in this project:

- ``y`` is 1 for a bad loan (charge-off/default) and 0 for a good one.
- ``score`` is oriented so that **higher means more likely bad** (a PD, or a log-odds of
  bad). Scorecard points, where higher means safer, must be negated before use.
- ``p`` arguments are probabilities of bad in [0, 1]; they are clipped to
  [1e-15, 1 - 1e-15] wherever a logit or log is taken.

AUC is the probability that a random bad scores above a random good (ties count half).
Gini = 2 AUC - 1. KS = max over thresholds of (TPR - FPR), which equals the two-sample
Kolmogorov-Smirnov statistic between the bad and good score distributions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

EPS = 1e-15


def _arrays(y: ArrayLike, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yv = np.asarray(y, dtype=np.float64)
    sv = np.asarray(s, dtype=np.float64)
    if yv.shape != sv.shape:
        raise ValueError(f"y and score shapes differ: {yv.shape} vs {sv.shape}")
    return yv, sv


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.clip(p, EPS, 1 - EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


# --- discrimination ------------------------------------------------------------------------


def auc(y: ArrayLike, score: ArrayLike) -> float:
    """Area under the ROC curve; 0.5 for a constant score."""
    yv, sv = _arrays(y, score)
    return float(roc_auc_score(yv, sv))


def gini(y: ArrayLike, score: ArrayLike) -> float:
    """Gini (accuracy ratio) = 2 AUC - 1."""
    return 2.0 * auc(y, score) - 1.0


def ks(y: ArrayLike, score: ArrayLike) -> float:
    """Kolmogorov-Smirnov statistic = max(TPR - FPR) over all thresholds (>= 0)."""
    yv, sv = _arrays(y, score)
    fpr, tpr, _ = roc_curve(yv, sv)
    return float(max(0.0, np.max(tpr - fpr)))


def pr_auc(y: ArrayLike, score: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision), bads as positives."""
    yv, sv = _arrays(y, score)
    return float(average_precision_score(yv, sv))


# --- probability quality -----------------------------------------------------------------------


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Mean squared error of the PD against the 0/1 outcome."""
    yv, pv = _arrays(y, p)
    return float(np.mean((pv - yv) ** 2))


def log_loss(y: ArrayLike, p: ArrayLike) -> float:
    """Mean negative log-likelihood in nats, PDs clipped away from 0 and 1."""
    yv, pv = _arrays(y, p)
    q = np.clip(pv, EPS, 1 - EPS)
    return float(-np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q)))


def _logistic_fit(
    y: NDArray[np.float64], x: NDArray[np.float64], offset: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Newton-Raphson for logit(P(y=1)) = x @ beta + offset."""
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        mu = _sigmoid(x @ beta + offset)
        w = mu * (1 - mu)
        step = np.linalg.solve((x * w[:, None]).T @ x, x.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_slope_intercept(y: ArrayLike, p: ArrayLike) -> tuple[float, float]:
    """(slope, intercept) on the log-odds scale.

    slope: ``b`` in ``logit P(bad) = a + b * logit(p)``; 1 is ideal, < 1 means the PDs are
    too extreme. intercept (calibration-in-the-large): ``a`` in
    ``logit P(bad) = a + logit(p)``; 0 is ideal, > 0 means the PDs are too low on average.
    """
    yv, pv = _arrays(y, p)
    z = _logit(pv)
    ones = np.ones_like(z)
    slope = _logistic_fit(yv, np.column_stack([ones, z]), np.zeros_like(z))[1]
    intercept = _logistic_fit(yv, ones[:, None], z)[0]
    return float(slope), float(intercept)


def calibration_table(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Quantile bins of the PD: ``bin`` (1 = lowest PD), ``n``, ``mean_pd``, ``observed_rate``."""
    yv, pv = _arrays(y, p)
    bins = pd.qcut(rankdata(pv, method="ordinal"), n_bins, labels=False) + 1
    df = pd.DataFrame({"bin": bins, "y": yv, "p": pv})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed_rate=("y", "mean"))
    return out.reset_index()


# --- comparison and uncertainty -------------------------------------------------------------------


@dataclass(frozen=True)
class DeLongResult:
    """DeLong test of AUC(a) - AUC(b) on the same rows (paired)."""

    auc_a: float
    auc_b: float
    delta: float
    se: float
    z: float
    p_value: float


def _delong_components(
    s: NDArray[np.float64], pos: NDArray[np.bool_]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    x, yneg = s[pos], s[~pos]
    m, n = len(x), len(yneg)
    tz = rankdata(np.r_[x, yneg])
    tx, ty = rankdata(x), rankdata(yneg)
    auc_val = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n  # per bad: share of goods it outranks
    v01 = 1.0 - (tz[m:] - ty) / m  # per good: share of bads that outrank it
    return float(auc_val), v10, v01


def delong_test(y: ArrayLike, score_a: ArrayLike, score_b: ArrayLike) -> DeLongResult:
    """Paired DeLong test (DeLong et al. 1988; midrank form of Sun & Xu 2014)."""
    yv, a = _arrays(y, score_a)
    _, b = _arrays(y, score_b)
    pos = yv == 1
    m, n = int(pos.sum()), int((~pos).sum())
    auc_a, v10_a, v01_a = _delong_components(a, pos)
    auc_b, v10_b, v01_b = _delong_components(b, pos)
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / m + s01 / n
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    delta = auc_a - auc_b
    if var <= 0:
        return DeLongResult(auc_a, auc_b, delta, 0.0, 0.0, 1.0)
    se = float(np.sqrt(var))
    z = delta / se
    return DeLongResult(auc_a, auc_b, delta, se, z, float(2 * norm.sf(abs(z))))


@dataclass(frozen=True)
class BootstrapResult:
    """Paired bootstrap of metric(a) - metric(b): point delta, SE, percentile CI."""

    delta: float
    se: float
    ci_low: float
    ci_high: float


def _resample_indices(
    y: NDArray[np.float64], n_resamples: int, seed: int
) -> Iterator[NDArray[np.int_]]:
    """Yield ``n_resamples`` bootstrap index arrays, one at a time (memory: one array)."""
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < n_resamples:
        idx = rng.integers(0, len(y), size=len(y))
        if 0 < y[idx].sum() < len(idx):  # skip the (rare) one-class resample
            produced += 1
            yield idx


def paired_bootstrap(
    y: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float],
    n_resamples: int = 2_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Resample rows with replacement, the same rows for both scores."""
    yv, a = _arrays(y, score_a)
    _, b = _arrays(y, score_b)
    deltas = np.array(
        [
            metric(yv[i], a[i]) - metric(yv[i], b[i])
            for i in _resample_indices(yv, n_resamples, seed)
        ]
    )
    low, high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        delta=metric(yv, a) - metric(yv, b),
        se=float(np.std(deltas, ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
    )


def bootstrap_ci(
    y: ArrayLike,
    score: ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float],
    n_resamples: int = 2_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point, low, high): percentile bootstrap CI of ``metric``."""
    yv, sv = _arrays(y, score)
    vals = np.array([metric(yv[i], sv[i]) for i in _resample_indices(yv, n_resamples, seed)])
    low, high = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return metric(yv, sv), float(low), float(high)

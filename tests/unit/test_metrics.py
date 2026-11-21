"""Step 2.1 (EV-01): metric library. T-10 (KS), T-11 (AUC invariances), calibration, DeLong."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import ks_2samp, norm
from sklearn.metrics import roc_curve

from credit_risk.evaluation.metrics import (
    auc,
    bootstrap_ci,
    brier,
    calibration_slope_intercept,
    calibration_table,
    delong_test,
    gini,
    ks,
    log_loss,
    paired_bootstrap,
    pr_auc,
)

RNG = np.random.default_rng(42)


def _sample(n: int = 5_000, shift: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = rng.normal(size=n) + shift * y
    return y, score


# --- T-10: KS --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t10_ks_equals_ks_2samp_and_max_tpr_minus_fpr(seed: int) -> None:
    y, score = _sample(seed=seed)
    fpr, tpr, _ = roc_curve(y, score)
    expected = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert ks(y, score) == pytest.approx(expected, abs=1e-12)
    assert ks(y, score) == pytest.approx(np.max(tpr - fpr), abs=1e-12)


def test_t10_ks_with_heavy_ties() -> None:
    y, score = _sample(seed=3)
    tied = np.round(score)  # a handful of distinct values
    assert ks(y, tied) == pytest.approx(ks_2samp(tied[y == 1], tied[y == 0]).statistic, abs=1e-12)


# --- T-11: AUC invariances --------------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), scale=st.floats(0.1, 10.0), offset=st.floats(-5.0, 5.0))
def test_t11_auc_invariant_to_monotone_transforms(seed: int, scale: float, offset: float) -> None:
    y, score = _sample(n=500, seed=seed)
    base = auc(y, score)
    assert auc(y, scale * score + offset) == pytest.approx(base, abs=1e-12)
    assert auc(y, np.exp(score / 4)) == pytest.approx(base, abs=1e-12)


def test_t11_gini_is_two_auc_minus_one() -> None:
    y, score = _sample()
    assert gini(y, score) == pytest.approx(2 * auc(y, score) - 1)


def test_t11_constant_score_has_auc_one_half() -> None:
    """M0 (the train bad rate for everyone) must score exactly 0.5."""
    y, _ = _sample()
    assert auc(y, np.full(len(y), 0.13)) == 0.5
    assert ks(y, np.full(len(y), 0.13)) == 0.0


def test_t11_orientation_higher_score_means_more_likely_bad() -> None:
    y, score = _sample(shift=2.0)
    assert auc(y, score) > 0.8
    assert auc(y, -score) == pytest.approx(1 - auc(y, score))


def test_closed_form_two_gaussians() -> None:
    """N(0,1) goods vs N(1,1) bads: AUC = Phi(1/sqrt 2), KS = 2 Phi(0.5) - 1 (T-05 part 6)."""
    n = 200_000
    y = np.r_[np.zeros(n), np.ones(n)]
    score = np.r_[RNG.normal(0, 1, n), RNG.normal(1, 1, n)]
    assert auc(y, score) == pytest.approx(norm.cdf(1 / np.sqrt(2)), abs=0.005)
    assert ks(y, score) == pytest.approx(2 * norm.cdf(0.5) - 1, abs=0.005)


# --- probability metrics ------------------------------------------------------------------------


def test_brier_and_log_loss_hand_values() -> None:
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.6, 0.5])
    assert brier(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.25) / 4)
    expected = -np.mean(np.log([0.9, 0.8, 0.6, 0.5]))
    assert log_loss(y, p) == pytest.approx(expected)


def test_log_loss_is_finite_at_0_and_1() -> None:
    assert np.isfinite(log_loss(np.array([0, 1]), np.array([1.0, 0.0])))


def test_pr_auc_of_perfect_ranking_is_one_and_random_is_base_rate() -> None:
    y, _ = _sample(n=50_000)
    assert pr_auc(y, y.astype(float)) == pytest.approx(1.0)
    assert pr_auc(y, RNG.random(len(y))) == pytest.approx(y.mean(), abs=0.02)


def _calibrated(n: int = 200_000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logit = rng.normal(-2.0, 1.0, size=n)
    p = 1 / (1 + np.exp(-logit))
    return rng.binomial(1, p), p


def test_calibration_slope_and_intercept_of_true_probabilities() -> None:
    y, p = _calibrated()
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.03)
    assert intercept == pytest.approx(0.0, abs=0.03)


def test_calibration_slope_detects_overconfidence() -> None:
    y, p = _calibrated()
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2 * logit))  # logits doubled: too extreme
    slope, _ = calibration_slope_intercept(y, over)
    assert slope == pytest.approx(0.5, abs=0.03)


def test_calibration_intercept_detects_a_level_shift() -> None:
    y, p = _calibrated()
    logit = np.log(p / (1 - p))
    low = 1 / (1 + np.exp(-(logit - 0.5)))  # every PD too low by 0.5 on the log-odds scale
    _, intercept = calibration_slope_intercept(y, low)
    assert intercept == pytest.approx(0.5, abs=0.03)


def test_calibration_table() -> None:
    y, p = _calibrated()
    table = calibration_table(y, p, n_bins=10)
    assert list(table.columns) == ["bin", "n", "mean_pd", "observed_rate"]
    assert len(table) == 10
    assert table["n"].sum() == len(y)
    assert table["mean_pd"].is_monotonic_increasing
    assert np.allclose(table["mean_pd"], table["observed_rate"], atol=0.01)


# --- DeLong and bootstrap -------------------------------------------------------------------------


def test_delong_detects_a_real_difference() -> None:
    y, strong = _sample(n=20_000, shift=1.0, seed=11)
    weak = strong + np.random.default_rng(12).normal(0, 2.0, size=len(y))
    res = delong_test(y, strong, weak)
    assert res.auc_a == pytest.approx(auc(y, strong))
    assert res.auc_b == pytest.approx(auc(y, weak))
    assert res.p_value < 1e-6


def test_delong_identical_scores_give_p_one() -> None:
    y, s = _sample(n=2_000)
    res = delong_test(y, s, s.copy())
    assert res.delta == 0.0
    assert res.p_value == 1.0


def test_delong_standard_error_matches_bootstrap() -> None:
    y, a = _sample(n=2_000, seed=21)
    b = a + np.random.default_rng(22).normal(0, 1.0, size=len(y))
    res = delong_test(y, a, b)
    boot = paired_bootstrap(y, a, b, auc, n_resamples=800, seed=42)
    assert res.se == pytest.approx(boot.se, rel=0.15)


def test_paired_bootstrap_is_seeded_and_brackets_the_point() -> None:
    y, a = _sample(n=2_000, seed=31)
    b = a + np.random.default_rng(32).normal(0, 1.0, size=len(y))
    r1 = paired_bootstrap(y, a, b, auc, n_resamples=200, seed=42)
    r2 = paired_bootstrap(y, a, b, auc, n_resamples=200, seed=42)
    assert r1 == r2
    assert r1.delta == pytest.approx(auc(y, a) - auc(y, b))
    assert r1.ci_low < r1.delta < r1.ci_high


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    y, s = _sample(n=3_000)
    point, low, high = bootstrap_ci(y, s, auc, n_resamples=500, seed=42)
    assert point == pytest.approx(auc(y, s))
    assert low < point < high


def test_bootstrap_resamples_are_generated_lazily() -> None:
    """2,000 index arrays of the 282k-row OOT set would need ~4.5 GB if built up front."""
    import inspect

    from credit_risk.evaluation.metrics import _resample_indices

    y = np.r_[np.zeros(50), np.ones(50)]
    assert inspect.isgenerator(_resample_indices(y, 10, seed=42))

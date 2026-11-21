"""T-05: every model and metric is checked against data whose answer is known (PLAN §9).

Parts 1, 3 (IV), 4 and 6 run now. Parts 2, 3 (selection), 5 and 7 need model code that
only exists after ``prereg-v1`` (Phase 2-3, 5). They are strict xfails: once the import
works and the assertion passes, pytest reports XPASS as a failure, which is the signal to
remove the marker and adjust the call to the real API.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TRUE_COEF, TRUE_INTERCEPT, Synthetic
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from credit_risk.data.audit import information_value
from credit_risk.evaluation.metrics import auc, calibration_slope_intercept, ks

FEATURES = ["x1", "x2", "x3", "noise"]
NEEDS_MODELS = "needs model code from Phase 2-3 (after prereg-v1)"


def _plain_lr(s: Synthetic) -> LogisticRegression:
    lr = LogisticRegression(C=np.inf, max_iter=1_000)  # unpenalised (sklearn >= 1.8 idiom)
    return lr.fit(s.dev[FEATURES], s.dev["y"])


def test_t05_bayes_auc_is_well_defined(synthetic: Synthetic) -> None:
    bayes = auc(synthetic.oot["y"], synthetic.oot["true_p"])
    assert 0.70 < bayes < 0.80  # sanity: the truth is informative but far from perfect


def test_t05_part1_plain_lr_recovers_the_coefficients(synthetic: Synthetic) -> None:
    lr = _plain_lr(synthetic)
    assert lr.intercept_[0] == pytest.approx(TRUE_INTERCEPT, abs=0.05)
    for name, coef in zip(FEATURES, lr.coef_[0], strict=True):
        assert coef == pytest.approx(TRUE_COEF[name], abs=0.05), name


def test_t05_part3_noise_column_has_iv_below_0_02(synthetic: Synthetic) -> None:
    dev = synthetic.dev
    assert information_value(dev["noise"], dev["y"], kind="numeric") < 0.02
    assert information_value(dev["x1"], dev["y"], kind="numeric") > 0.1


def test_t05_part4_lr_calibration_slope_in_0_95_to_1_05(synthetic: Synthetic) -> None:
    p = _plain_lr(synthetic).predict_proba(synthetic.oot[FEATURES])[:, 1]
    slope, _ = calibration_slope_intercept(synthetic.oot["y"], p)
    assert 0.95 <= slope <= 1.05


def test_t05_part6_closed_form_two_gaussians() -> None:
    rng = np.random.default_rng(42)
    n = 200_000
    y = np.r_[np.zeros(n), np.ones(n)]
    score = np.r_[rng.normal(0, 1, n), rng.normal(1, 1, n)]
    assert auc(y, score) == pytest.approx(norm.cdf(1 / np.sqrt(2)), abs=0.005)  # 0.760
    assert ks(y, score) == pytest.approx(2 * norm.cdf(0.5) - 1, abs=0.005)  # 0.383


@pytest.mark.xfail(raises=ImportError, strict=True, reason=NEEDS_MODELS)
def test_t05_part2_scorecard_auc_within_0_01_of_bayes(synthetic: Synthetic) -> None:
    from credit_risk.models.scorecard import fit_scorecard  # type: ignore[import-not-found]

    model = fit_scorecard(synthetic.dev[FEATURES], synthetic.dev["y"])
    pd_oot = model.predict_proba(synthetic.oot[FEATURES])
    bayes = auc(synthetic.oot["y"], synthetic.oot["true_p"])
    assert auc(synthetic.oot["y"], pd_oot) == pytest.approx(bayes, abs=0.01)


@pytest.mark.xfail(raises=ImportError, strict=True, reason=NEEDS_MODELS)
def test_t05_part3_scorecard_drops_the_noise_column(synthetic: Synthetic) -> None:
    from credit_risk.models.scorecard import fit_scorecard  # type: ignore[import-not-found]

    model = fit_scorecard(synthetic.dev[FEATURES], synthetic.dev["y"])
    assert "noise" not in model.selected_features


@pytest.mark.xfail(raises=ImportError, strict=True, reason=NEEDS_MODELS)
def test_t05_part5_monotone_lightgbm_has_no_violations_in_x1(synthetic: Synthetic) -> None:
    from credit_risk.models.lgbm import fit_monotone_lgbm  # type: ignore[import-not-found]

    model = fit_monotone_lgbm(synthetic.dev[FEATURES], synthetic.dev["y"], {"x1": 1})
    row = synthetic.oot[FEATURES].iloc[[0] * 101].copy()
    row["x1"] = np.linspace(-3, 3, 101)
    assert np.all(np.diff(model.predict_proba(row)) >= 0)


@pytest.mark.xfail(raises=ImportError, strict=True, reason=NEEDS_MODELS)
def test_t05_part7_x1_moves_pd_more_than_noise_in_99pct_of_rows(synthetic: Synthetic) -> None:
    from credit_risk.models.scorecard import fit_scorecard  # type: ignore[import-not-found]

    model = fit_scorecard(synthetic.dev[FEATURES], synthetic.dev["y"])
    x = synthetic.oot[FEATURES].iloc[:2_000]
    base = model.predict_proba(x)
    moved = {}
    for col in ("x1", "noise"):
        shifted = x.copy()
        shifted[col] = shifted[col] + 1.0
        moved[col] = np.abs(model.predict_proba(shifted) - base)
    assert np.mean(moved["x1"] > moved["noise"]) >= 0.99

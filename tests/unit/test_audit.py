"""Step 1.4 (D-05): feature leakage and missingness audit (train window only)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FeatureSpec, MonthRange
from credit_risk.data.audit import (
    feature_audit,
    information_value,
    l6_flag,
    missing_by_target,
    render_audit_md,
)

TRAIN = MonthRange(start="2010-01", end="2011-12")


def test_iv_hand_computed_two_categories() -> None:
    x = pd.Series(["A"] * 100 + ["B"] * 100)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    # each bin: (0.8 - 0.2) * ln(0.8 / 0.2) = 0.6 ln 4
    assert information_value(x, y, kind="categorical") == pytest.approx(2 * 0.6 * math.log(4))


def test_iv_of_an_independent_feature_is_near_zero() -> None:
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(size=50_000))
    y = pd.Series(rng.integers(0, 2, size=50_000))
    assert information_value(x, y, kind="numeric") < 0.01


def test_iv_treats_missing_as_its_own_bin_and_stays_finite() -> None:
    x = pd.Series([1.0] * 50 + [np.nan] * 50)
    y = pd.Series([0] * 50 + [1] * 50)
    iv = information_value(x, y, kind="numeric")
    assert math.isfinite(iv)
    assert iv > 2.0


def test_missing_by_target() -> None:
    x = pd.Series([np.nan, 1.0, 1.0, 1.0, np.nan, np.nan])
    y = pd.Series([0, 0, 0, 0, 1, 1])
    assert missing_by_target(x, y) == (pytest.approx(0.25), pytest.approx(1.0))


@pytest.mark.parametrize(
    ("good", "bad", "flag"),
    [
        (0.10, 0.40, True),  # 4x
        (0.40, 0.10, True),  # direction does not matter
        (0.10, 0.25, False),  # 2.5x
        (0.0, 0.005, False),  # below the 1% materiality floor
        (0.0, 0.05, True),
        (0.0, 0.0, False),
    ],
)
def test_l6_flag(good: float, bad: float, flag: bool) -> None:
    assert l6_flag(good, bad) is flag


def _table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4_000
    months = pd.to_datetime(rng.choice(["2010-03-01", "2011-06-01", "2012-05-01"], size=n))
    y = rng.integers(0, 2, size=n)
    in_valid = months.year == 2012
    df = pd.DataFrame(
        {
            "issue_d": months,
            "y": pd.array(y, dtype="Int8"),
            "segment": "core",
            # perfect predictor ONLY in the 2012 (non-train) rows: must not show up
            "peeky": np.where(in_valid, y, rng.normal(size=n)),
            "const": "36 months",
            "leaky_missing": np.where(y == 1, np.nan, 1.0),
        }
    )
    return df


SPECS = [
    FeatureSpec(name="peeky", kind="numeric"),
    FeatureSpec(name="const", kind="categorical"),
    FeatureSpec(name="leaky_missing", kind="numeric"),
]


def test_feature_audit_uses_train_rows_only_and_decides_every_feature() -> None:
    audit, by_year = feature_audit(_table(), SPECS, TRAIN, iv_min=0.02, iv_leakage_flag=0.5)
    audit = audit.set_index("feature")
    assert list(audit.index) == ["peeky", "const", "leaky_missing"]
    assert set(audit["decision"]) <= {"keep", "drop"}
    assert audit.loc["peeky", "iv_train"] < 0.05  # 2012 rows were not used
    assert audit.loc["peeky", "decision"] == "keep"
    assert audit.loc["const", "decision"] == "drop"
    assert "constant" in audit.loc["const", "reason"]
    assert audit.loc["leaky_missing", "l6_flag"]
    assert audit.loc["leaky_missing", "decision"] == "drop"
    assert sorted(by_year["year"].unique()) == [2010, 2011]


def test_feature_audit_drops_iv_above_leakage_flag() -> None:
    df = _table()
    df["strong"] = np.where(df["y"] == 1, 1.0, 0.0)
    specs = [FeatureSpec(name="strong", kind="numeric")]
    audit, _ = feature_audit(df, specs, TRAIN, iv_min=0.02, iv_leakage_flag=0.5)
    row = audit.iloc[0]
    assert row["decision"] == "drop"
    assert "leakage flag" in row["reason"]


def test_render_audit_md_has_a_row_per_feature() -> None:
    audit, by_year = feature_audit(_table(), SPECS, TRAIN, iv_min=0.02, iv_leakage_flag=0.5)
    md = render_audit_md(audit, by_year)
    assert md.startswith("# Feature audit")
    for name in ("peeky", "const", "leaky_missing"):
        assert f"| {name} |" in md

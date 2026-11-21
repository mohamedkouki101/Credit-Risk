"""Step 1.3 (D-04): derived features, T-01 (whitelist + leakage regex), T-04 (date arithmetic)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FeaturesConfig, load_all_configs
from credit_risk.data.features import (
    LEAKAGE_REGEX,
    LeakageError,
    check_model_columns,
    derive,
    model_frame,
    months_between,
)

FEATURES: FeaturesConfig = load_all_configs().features
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lc_accepted_columns.txt"
ALL_RAW = [
    line.strip()
    for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]
PLAN_REGEX = r"pymnt|rec_|recover|settle|hardship|out_prncp|last_"  # data plan 5.6, L1


def _clean_rows() -> pd.DataFrame:
    """Two cleaned rows with every column ``derive`` needs."""
    return pd.DataFrame(
        {
            "id": ["1", "2"],
            "issue_d": pd.to_datetime(["2015-03-01", "2012-02-01"]),
            "earliest_cr_line": pd.to_datetime(["2000-02-01", "2008-02-01"]),
            "loan_amnt": [1000.0, 2000.0],
            "purpose": ["credit_card", "car"],
            "term": ["36 months", "36 months"],
            "emp_length": ["< 1 year", "10+ years"],
            "home_ownership": ["ANY", "MORTGAGE"],
            "annual_inc": [0.0, math.e - 1],
            "verification_status": ["Verified", "Not Verified"],
            "application_type": ["Individual", "Individual"],
            "dti": [10.0, np.nan],
            "fico_range_low": [700.0, 660.0],
            "fico_range_high": [704.0, 664.0],
            "inq_last_6mths": [0.0, 1.0],
            "delinq_2yrs": [0.0, 0.0],
            "mths_since_last_delinq": [np.nan, 12.0],
            "mths_since_last_record": [np.nan, np.nan],
            "open_acc": [5.0, 7.0],
            "total_acc": [10.0, 20.0],
            "pub_rec": [0.0, 1.0],
            "pub_rec_bankruptcies": [0.0, 1.0],
            "revol_bal": [0.0, 99.0],
            "revol_util": [10.0, 50.0],
            "mort_acc": [0.0, 2.0],
            "sub_grade": ["C4", "A1"],
            "zip_code": ["190xx", "940xx"],
            "loan_status": ["Fully Paid", "Charged Off"],
            "total_rec_prncp": [1000.0, 300.0],
            "y": pd.array([0, 1], dtype="Int8"),
        }
    )


# --- T-04: credit_age_months date arithmetic ----------------------------------------------


@pytest.mark.parametrize(
    ("issue", "earliest", "expected"),
    [
        ("2015-03-01", "2000-02-01", 181.0),  # 15 years + 1 month
        ("2012-02-01", "2008-02-01", 48.0),  # February, leap years on both ends
        ("2012-02-29", "2008-02-01", 48.0),  # day of month is ignored: whole months only
        ("2016-02-01", "2015-12-01", 2.0),  # across a year boundary
        ("2010-01-01", "2010-01-01", 0.0),  # same month
    ],
)
def test_t04_months_between(issue: str, earliest: str, expected: float) -> None:
    out = months_between(pd.Series(pd.to_datetime([issue])), pd.Series(pd.to_datetime([earliest])))
    assert out.iloc[0] == expected


def test_t04_credit_age_uses_issue_date_not_today() -> None:
    out = derive(_clean_rows())
    assert out["credit_age_months"].tolist() == [181.0, 48.0]


def test_t04_negative_or_unknown_credit_age_is_missing() -> None:
    df = _clean_rows()
    df["earliest_cr_line"] = pd.to_datetime(["2016-01-01", None])
    out = derive(df)
    assert out["credit_age_months"].isna().all()


# --- other derivations ------------------------------------------------------------------


def test_fico_mid_is_the_mean_of_the_range() -> None:
    assert derive(_clean_rows())["fico_mid"].tolist() == [702.0, 662.0]


def test_emp_length_is_ordinal_0_to_10_and_null_stays_missing() -> None:
    df = _clean_rows()
    assert derive(df)["emp_length"].tolist() == [0.0, 10.0]
    df["emp_length"] = ["5 years", None]
    out = derive(df)
    assert out["emp_length"].iloc[0] == 5.0
    assert pd.isna(out["emp_length"].iloc[1])


def test_unknown_emp_length_raises() -> None:
    df = _clean_rows()
    df["emp_length"] = ["n/a", "1 year"]
    with pytest.raises(ValueError, match="emp_length"):
        derive(df)


def test_home_ownership_rare_values_merge_into_other() -> None:
    df = _clean_rows()
    df["home_ownership"] = ["NONE", "ANY"]
    assert derive(df)["home_ownership"].tolist() == ["OTHER", "OTHER"]
    assert derive(_clean_rows())["home_ownership"].tolist() == ["OTHER", "MORTGAGE"]


def test_income_and_revolving_balance_are_log1p() -> None:
    out = derive(_clean_rows())
    assert out["annual_inc"].tolist() == pytest.approx([0.0, 1.0])
    assert out["revol_bal"].tolist() == pytest.approx([0.0, math.log(100.0)])


def test_derive_does_not_mutate_input() -> None:
    df = _clean_rows()
    before = df.copy()
    derive(df)
    pd.testing.assert_frame_equal(df, before)


# --- T-01: whitelist and leakage regex ----------------------------------------------------


def test_t01_model_frame_is_exactly_the_whitelist() -> None:
    x = model_frame(derive(_clean_rows()), FEATURES)
    assert list(x.columns) == FEATURES.feature_names()
    assert set(x.columns).isdisjoint(
        set(FEATURES.benchmark_only) | set(FEATURES.fairness_only) | set(FEATURES.outcome_only)
    )
    assert "y" not in x.columns


def test_t01_no_whitelist_feature_matches_the_leakage_regex() -> None:
    names = FEATURES.feature_names() + sorted(FEATURES.feature_source_columns())
    assert [n for n in names if re.search(LEAKAGE_REGEX, n)] == []


def test_t01_regex_still_catches_every_post_origination_column() -> None:
    """The plan's regex, narrowed only where it hit application-time bureau counts."""
    plan_hits = {c for c in ALL_RAW if re.search(PLAN_REGEX, c)}
    our_hits = {c for c in ALL_RAW if re.search(LEAKAGE_REGEX, c)}
    assert our_hits <= plan_hits
    assert plan_hits - our_hits == {
        "inq_last_6mths",
        "inq_last_12m",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "pub_rec_bankruptcies",
        "sec_app_inq_last_6mths",
        "sec_app_mths_since_last_major_derog",
    }


@pytest.mark.parametrize(
    "column",
    ["sub_grade", "int_rate", "zip_code", "addr_state", "loan_status", "total_rec_prncp", "y"],
)
def test_t01_restricted_columns_are_rejected(column: str) -> None:
    with pytest.raises(LeakageError, match=column):
        check_model_columns([*FEATURES.feature_names(), column], FEATURES)


@pytest.mark.parametrize("column", ["last_pymnt_amnt", "total_pymnt", "hardship_flag"])
def test_t01_post_origination_columns_are_rejected(column: str) -> None:
    with pytest.raises(LeakageError, match=column):
        check_model_columns([column], FEATURES)


def test_t01_unlisted_column_is_rejected_by_default() -> None:
    with pytest.raises(LeakageError, match="not in the whitelist"):
        check_model_columns(["fico_mid", "member_id"], FEATURES)

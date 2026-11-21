"""Step 1.2 (D-03): target construction (data plan 5.3a), T-06, and the vintage profile."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.config import TargetConfig, load_all_configs
from credit_risk.data.target import build_target, render_profile_md, vintage_profile

CFG: TargetConfig = load_all_configs().target


def _frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows: (loan_status, application_type, term, issue month 'YYYY-MM')."""
    df = pd.DataFrame(rows, columns=["loan_status", "application_type", "term", "issue_d"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%Y-%m")
    return df


def _core(status: str, month: str = "2012-03") -> tuple[str, str, str, str]:
    return (status, "Individual", "36 months", month)


# --- T-06: the status mapping is exhaustive ----------------------------------------------


def test_t06_unknown_status_raises() -> None:
    df = _frame([_core("Fully Paid"), _core("Charged Off (legacy)")])
    with pytest.raises(ValueError, match="Charged Off \\(legacy\\)"):
        build_target(df, CFG)


def test_t06_null_status_raises() -> None:
    df = _frame([_core("Fully Paid")])
    df.loc[0, "loan_status"] = None
    with pytest.raises(ValueError, match="null"):
        build_target(df, CFG)


def test_t06_every_configured_status_maps_without_error() -> None:
    statuses = list(CFG.status_to_class())
    assert len(statuses) == 9
    out = build_target(_frame([_core(s) for s in statuses]), CFG)
    assert len(out) == 9


# --- mapping and population ---------------------------------------------------------------


def test_bad_good_indeterminate_mapping() -> None:
    out = build_target(
        _frame([_core("Charged Off"), _core("Default"), _core("Fully Paid"), _core("Current")]),
        CFG,
    )
    assert out["y"].tolist()[:3] == [1, 1, 0]
    assert pd.isna(out.loc[3, "y"])
    assert out["exclusion_reason"].tolist()[3] == "indeterminate"
    assert out["exclusion_reason"].iloc[:3].isna().all()
    assert (out["segment"] == "core").all()


def test_policy_loans_are_kept_with_a_flag() -> None:
    out = build_target(
        _frame(
            [
                _core("Does not meet the credit policy. Status:Charged Off", "2008-01"),
                _core("Does not meet the credit policy. Status:Fully Paid", "2008-01"),
                _core("Fully Paid", "2008-01"),
            ]
        ),
        CFG,
    )
    assert out["y"].tolist() == [1, 0, 0]
    assert out["policy_flag"].tolist() == [True, True, False]


def test_non_individual_and_60_month_loans_are_out_of_population() -> None:
    out = build_target(
        _frame(
            [
                ("Charged Off", "Joint App", "36 months", "2015-06"),
                ("Charged Off", "Individual", "60 months", "2015-06"),
            ]
        ),
        CFG,
    )
    assert out["exclusion_reason"].tolist() == ["application_type", "term"]
    assert out["y"].isna().all()
    assert (out["segment"] == "out_of_population").all()


def test_issue_window_boundaries_are_inclusive() -> None:
    months = ["2007-05", "2007-06", "2015-12", "2016-01"]
    out = build_target(_frame([_core("Fully Paid", m) for m in months]), CFG)
    assert out["exclusion_reason"].tolist() == ["issue_d_window", None, None, "issue_d_window"]
    assert out["y"].notna().tolist() == [False, True, True, False]


def test_input_is_not_mutated_and_index_is_kept() -> None:
    df = _frame([_core("Fully Paid"), _core("Charged Off")]).set_index(pd.Index([10, 20]))
    before = df.copy()
    out = build_target(df, CFG)
    pd.testing.assert_frame_equal(df, before)
    assert out.index.tolist() == [10, 20]
    assert str(out["y"].dtype) == "Int8"


def test_unparsed_issue_date_raises_type_error() -> None:
    df = _frame([_core("Fully Paid")])
    df["issue_d"] = "Mar-2012"
    with pytest.raises(TypeError, match="issue_d"):
        build_target(df, CFG)


# --- vintage profile -----------------------------------------------------------------------


def _profile_input() -> pd.DataFrame:
    df = _frame(
        [
            _core("Charged Off", "2010-01"),
            _core("Fully Paid", "2010-02"),
            _core("Current", "2010-03"),
            _core("Fully Paid", "2010-04"),
            _core("Fully Paid", "2010-05"),
            ("Charged Off", "Individual", "60 months", "2010-05"),  # not in the profile
            _core("Current", "2016-02"),  # outside the window: not in the profile
        ]
    )
    df["dti"] = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0]
    df["fico_range_low"] = [700.0, 710.0, None, 720.0, 730.0, 740.0, 750.0]
    return build_target(df, CFG)


def test_vintage_profile_counts_per_issue_quarter() -> None:
    prof = vintage_profile(_profile_input(), ["dti", "fico_range_low"], flag_share=0.01)
    assert prof["quarter"].tolist() == ["2010Q1", "2010Q2"]
    q1, q2 = prof.iloc[0], prof.iloc[1]
    assert (q1["rows"], q1["n_bad"], q1["n_good"], q1["n_indeterminate"]) == (3, 1, 1, 1)
    assert q1["bad_rate"] == pytest.approx(0.5)
    assert q1["indeterminate_share"] == pytest.approx(1 / 3)
    assert bool(q1["flagged"]) is True
    assert q1["missing_share"] == pytest.approx(2 / 6)
    assert (q2["rows"], q2["n_bad"], q2["n_good"]) == (2, 0, 2)
    assert q2["bad_rate"] == pytest.approx(0.0)
    assert bool(q2["flagged"]) is False
    assert q2["missing_share"] == pytest.approx(0.0)


def test_render_profile_md_lists_flagged_vintages() -> None:
    prof = vintage_profile(_profile_input(), ["dti", "fico_range_low"], flag_share=0.01)
    md = render_profile_md(prof, flag_share=0.01)
    assert md.startswith("# Data profile")
    assert "| 2010Q1 |" in md
    assert "Flagged vintages (indeterminate share > 1.0%): 2010Q1" in md

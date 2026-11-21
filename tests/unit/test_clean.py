"""Step 1.3 (D-04): cleaning rules of data plan 5.4, each with a test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.clean import clean


def _raw(**overrides: list[object]) -> pd.DataFrame:
    """Three valid raw rows; ``overrides`` replaces whole columns."""
    cols: dict[str, list[object]] = {
        "id": ["1", "2", "3"],
        "issue_d": ["Mar-2012", "Feb-2012", "Dec-2015"],
        "earliest_cr_line": ["Aug-2003", "Feb-2008", "Jan-1999"],
        "term": [" 36 months", " 60 months", " 36 months"],
        "loan_amnt": [1000.0, 2500.0, 3600.0],
        "dti": [5.0, 100.0, 0.0],
        "revol_util": [29.7, 150.0, 0.0],
        "int_rate": [13.99, 7.5, 11.0],
        "annual_inc": [55000.0, 0.0, 120000.0],
    }
    cols.update(overrides)
    return pd.DataFrame(cols)


def test_valid_rows_pass_through_with_parsed_dates_and_stripped_term() -> None:
    out, log = clean(_raw())
    assert len(out) == 3
    assert log.total_dropped == 0
    assert out["issue_d"].tolist() == [
        pd.Timestamp("2012-03-01"),
        pd.Timestamp("2012-02-01"),
        pd.Timestamp("2015-12-01"),
    ]
    assert out["earliest_cr_line"].iloc[1] == pd.Timestamp("2008-02-01")
    assert out["term"].tolist() == ["36 months", "60 months", "36 months"]


def test_rule4_footer_rows_with_empty_loan_amnt_are_dropped_and_counted() -> None:
    raw = _raw(
        id=["1", "2", "Total amount funded in policy code 1: 99"],
        issue_d=["Mar-2012", "Feb-2012", None],
        loan_amnt=[1000.0, 2500.0, np.nan],
    )
    out, log = clean(raw)
    assert out["id"].tolist() == ["1", "2"]
    assert log.drops["footer_empty_loan_amnt"] == 1
    assert log.drops["issue_d_unparsed"] == 0  # the footer row is not double-counted


def test_rule1_unparsable_dates_are_dropped_and_counted() -> None:
    out, log = clean(_raw(issue_d=["Mar-2012", "2012-02", "Dec-2015"]))
    assert out["id"].tolist() == ["1", "3"]
    assert log.drops["issue_d_unparsed"] == 1

    out, log = clean(_raw(earliest_cr_line=["Aug-2003", "Foo-2001", "Jan-1999"]))
    assert out["id"].tolist() == ["1", "3"]
    assert log.drops["earliest_cr_line_unparsed"] == 1


def test_rule1_null_earliest_cr_line_is_kept_as_missing() -> None:
    out, log = clean(_raw(earliest_cr_line=["Aug-2003", None, "Jan-1999"]))
    assert len(out) == 3
    assert pd.isna(out["earliest_cr_line"].iloc[1])
    assert log.drops["earliest_cr_line_unparsed"] == 0


def test_rule2_percent_signs_are_stripped() -> None:
    out, _ = clean(_raw(revol_util=["29.7%", " 150% ", None], int_rate=["13.99%", "7.5%", "11%"]))
    assert out["revol_util"].iloc[:2].tolist() == [29.7, 150.0]
    assert pd.isna(out["revol_util"].iloc[2])
    assert out["int_rate"].tolist() == [13.99, 7.5, 11.0]
    assert out["revol_util"].dtype == "float64"


def test_rule3_duplicate_ids_are_dropped_and_counted() -> None:
    out, log = clean(_raw(id=["1", "1", "3"]))
    assert out["id"].tolist() == ["1", "3"]
    assert out["id"].is_unique
    assert log.drops["duplicate_id"] == 1


def test_rule5_out_of_range_dti_becomes_missing_with_flag() -> None:
    out, log = clean(_raw(dti=[-1.0, 100.0, 120.0]))
    assert pd.isna(out["dti"].iloc[0])
    assert out["dti"].iloc[1] == 100.0
    assert pd.isna(out["dti"].iloc[2])
    assert out["dti_flag"].tolist() == [True, False, True]
    assert log.flags["dti_out_of_range"] == 2


def test_revol_util_above_150_becomes_missing() -> None:
    out, log = clean(_raw(revol_util=[150.0, 150.1, 892.3]))
    assert out["revol_util"].iloc[0] == 150.0
    assert out["revol_util"].iloc[1:].isna().all()
    assert log.flags["revol_util_over_150"] == 2


def test_row_count_identity_and_input_not_mutated() -> None:
    raw = _raw(
        id=["1", "1", "x"],
        issue_d=["Mar-2012", "Feb-2012", None],
        loan_amnt=[1000.0, 2500.0, np.nan],
    )
    before = raw.copy()
    out, log = clean(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert log.raw_rows == 3
    assert len(out) == log.raw_rows - log.total_dropped == log.clean_rows
    assert out.index.tolist() == list(range(len(out)))


def test_cleaning_log_table_lists_every_rule() -> None:
    _, log = clean(_raw())
    table = log.to_frame()
    assert list(table.columns) == ["rule", "kind", "rows"]
    assert set(table["rule"]) >= {
        "footer_empty_loan_amnt",
        "issue_d_unparsed",
        "earliest_cr_line_unparsed",
        "duplicate_id",
        "dti_out_of_range",
        "revol_util_over_150",
    }


@pytest.mark.parametrize("bad", ["annual_inc", "loan_amnt"])
def test_impossible_values_are_left_for_the_schema_to_reject(bad: str) -> None:
    """Rule 5 range checks on these columns are pandera's job (schema.py), not silent fixes."""
    raw = _raw(**{bad: [-5.0, 1.0, 2.0]})
    out, _ = clean(raw)
    assert out[bad].iloc[0] == -5.0

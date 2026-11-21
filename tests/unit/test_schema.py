"""Step 1.3 (D-04): the pandera schema rejects impossible values (rule 5.4.5)."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest
from test_features import _clean_rows

from credit_risk.config import load_all_configs
from credit_risk.data.features import derive
from credit_risk.data.schema import LoanSchema
from credit_risk.data.target import build_target


def _valid() -> pd.DataFrame:
    df = _clean_rows().drop(columns=["y"])
    df["dti_flag"] = False
    return derive(build_target(df, load_all_configs().target))


def test_valid_frame_passes() -> None:
    LoanSchema.validate(_valid())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("fico_mid", 900.0),
        ("fico_mid", 250.0),
        ("loan_amnt", 0.0),
        ("annual_inc", -0.5),
        ("dti", 120.0),
        ("revol_util", 151.0),
        ("credit_age_months", -1.0),
        ("emp_length", 11.0),
        ("home_ownership", "ANY"),
        ("term", "48 months"),
        ("segment", "other"),
    ],
)
def test_out_of_range_values_fail(column: str, value: object) -> None:
    df = _valid()
    df.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaError):
        LoanSchema.validate(df)


def test_duplicate_id_fails() -> None:
    df = _valid()
    df.loc[1, "id"] = df.loc[0, "id"]
    with pytest.raises(pandera.errors.SchemaError):
        LoanSchema.validate(df)


def test_target_outside_0_1_fails() -> None:
    df = _valid()
    df["y"] = pd.array([0, 2], dtype="Int8")
    with pytest.raises(pandera.errors.SchemaError):
        LoanSchema.validate(df)


def test_unparsed_issue_date_fails() -> None:
    df = _valid()
    df["issue_d"] = ["Mar-2015", "Feb-2012"]
    with pytest.raises(pandera.errors.SchemaError):
        LoanSchema.validate(df)


def test_missing_earliest_cr_line_is_allowed() -> None:
    """Real file: some rows have no earliest_cr_line; credit age is then missing (rule 5.4.1)."""
    df = _valid()
    df.loc[0, "earliest_cr_line"] = pd.NaT
    df.loc[0, "credit_age_months"] = float("nan")
    LoanSchema.validate(df)

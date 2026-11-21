"""Pandera schema of the cleaned, targeted, derived loan table (D-04, rule 5.4.5).

Validated before ``data/interim/loans_clean.parquet`` is written. Ranges follow data plan
5.4 rule 5; ``annual_inc`` and ``revol_bal`` are already ``log1p`` here, so ``>= 0`` on them
means the raw value was ``>= 0``. Missing values are allowed wherever the plan keeps them
as their own bin. Extra columns are allowed (``strict=False``); the model input is guarded
separately by ``features.model_frame`` (T-01).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from credit_risk.data.features import EMP_LENGTH_YEARS


def _is_datetime(s: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(s))


def _num(*checks: pa.Check, nullable: bool = True) -> pa.Column:
    return pa.Column(float, checks=list(checks), nullable=nullable)


LoanSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(nullable=False, unique=True),
        "issue_d": pa.Column(checks=pa.Check(_is_datetime, name="is_datetime"), nullable=False),
        "earliest_cr_line": pa.Column(
            checks=pa.Check(_is_datetime, name="is_datetime"), nullable=True
        ),
        "loan_status": pa.Column(nullable=False),
        "term": pa.Column(checks=pa.Check.isin(["36 months", "60 months"]), nullable=False),
        "home_ownership": pa.Column(checks=pa.Check.isin(["MORTGAGE", "RENT", "OWN", "OTHER"])),
        "loan_amnt": _num(pa.Check.gt(0), nullable=False),
        "fico_mid": _num(pa.Check.in_range(300, 850)),
        "annual_inc": _num(pa.Check.ge(0)),
        "revol_bal": _num(pa.Check.ge(0)),
        "dti": _num(pa.Check.in_range(0, 100)),
        "revol_util": _num(pa.Check.in_range(0, 150)),
        "credit_age_months": _num(pa.Check.ge(0)),
        "emp_length": _num(pa.Check.in_range(0, max(EMP_LENGTH_YEARS.values()))),
        "y": pa.Column("Int8", checks=pa.Check.isin([0, 1]), nullable=True),
        "segment": pa.Column(checks=pa.Check.isin(["core", "out_of_population"])),
        "policy_flag": pa.Column(bool),
        "dti_flag": pa.Column(bool),
    },
    strict=False,
    name="LoanSchema",
)

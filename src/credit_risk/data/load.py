"""Read the raw Lending Club file with only the columns the project may use (D-02).

The column list comes from ``configs/features.yaml``: key columns, the raw sources of the
whitelisted features, and the three audit-only groups (incumbent benchmark, fairness,
outcome). Every other column, including all post-origination fields, is never read into
memory, so it cannot leak into ``X`` by accident.

Every column has an explicit dtype. Text in a numeric column raises instead of silently
turning the column into strings. Values are not cleaned here (no whitespace stripping,
no date parsing); that is ``clean.py`` (D-04).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from credit_risk.config import FeaturesConfig, load_all_configs

RawDtype = Literal["float64", "str"]

RAW_DTYPES: dict[str, RawDtype] = {
    # keys and dates (parsed with format %b-%Y in clean.py)
    "id": "str",  # numeric in real rows, text in the footer rows dropped by cleaning
    "issue_d": "str",
    "earliest_cr_line": "str",
    "last_pymnt_d": "str",
    # categoricals
    "purpose": "str",
    "term": "str",
    "emp_length": "str",
    "home_ownership": "str",
    "verification_status": "str",
    "application_type": "str",
    "grade": "str",
    "sub_grade": "str",
    "addr_state": "str",
    "zip_code": "str",
    "loan_status": "str",
    # numerics (integers with missing values are read as float64)
    "loan_amnt": "float64",
    "funded_amnt": "float64",
    "annual_inc": "float64",
    "dti": "float64",
    "fico_range_low": "float64",
    "fico_range_high": "float64",
    "inq_last_6mths": "float64",
    "delinq_2yrs": "float64",
    "mths_since_last_delinq": "float64",
    "mths_since_last_record": "float64",
    "open_acc": "float64",
    "total_acc": "float64",
    "pub_rec": "float64",
    "pub_rec_bankruptcies": "float64",
    "revol_bal": "float64",
    "revol_util": "float64",  # percent, already numeric in Kaggle v3 (no "%")
    "mort_acc": "float64",
    "int_rate": "float64",  # percent, already numeric in Kaggle v3 (no "%")
    "installment": "float64",
    "total_rec_prncp": "float64",
    "recoveries": "float64",
    "collection_recovery_fee": "float64",
}


def raw_columns(cfg: FeaturesConfig) -> list[str]:
    """Columns to read, in a stable order: keys, feature sources, benchmark, fairness, outcome."""
    ordered = [
        *cfg.key_columns,
        *(src for spec in cfg.features for src in spec.source_columns()),
        *cfg.benchmark_only,
        *cfg.fairness_only,
        *cfg.outcome_only,
    ]
    return list(dict.fromkeys(ordered))


def load_raw(
    path: Path, cfg: FeaturesConfig | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Read ``path`` (CSV, optionally gzipped) with only the allowed columns.

    Columns come back in ``raw_columns(cfg)`` order. ``cfg`` defaults to
    ``configs/features.yaml``. Raises ``KeyError`` if a column has no entry in ``RAW_DTYPES``
    and ``ValueError`` if a numeric column contains text.
    """
    cfg = cfg or load_all_configs().features
    cols = raw_columns(cfg)
    missing = [c for c in cols if c not in RAW_DTYPES]
    if missing:
        raise KeyError(f"no explicit dtype for raw column(s) {missing}")
    df = pd.read_csv(
        path,
        usecols=cols,
        dtype={c: RAW_DTYPES[c] for c in cols},
        nrows=nrows,
        low_memory=False,
    )
    return df[cols]

"""Derived features and the model-input guard (D-04, T-01, T-04).

``derive`` turns a cleaned frame into model-ready columns (data plan 5.2):

- ``fico_mid`` = mean of ``fico_range_low`` and ``fico_range_high``.
- ``credit_age_months`` = whole months from ``earliest_cr_line`` to **``issue_d``** (never
  the snapshot date, leakage check L8); negative or unknown -> missing.
- ``emp_length``: ``"< 1 year"`` = 0 ... ``"10+ years"`` = 10, null -> missing; any other
  text raises.
- ``home_ownership``: OTHER/NONE/ANY -> OTHER.
- ``annual_inc`` and ``revol_bal`` replaced by ``log1p`` of themselves.

Category merging that depends on shares (``purpose`` under 1% -> other) is fitted on train
only, inside the model pipeline, not here.

``model_frame`` is the only way to build ``X``: it selects the whitelist and runs
``check_model_columns``, which raises ``LeakageError`` for any column that is not
whitelisted, belongs to a restricted group, or matches ``LEAKAGE_REGEX``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from credit_risk.config import FeaturesConfig

# Data plan 5.6 L1, narrowed (see research/decisions.md): "last_" only as a prefix and
# "rec_" not after "pub_", so the application-time counts inq_last_6mths,
# mths_since_last_delinq/record and pub_rec_bankruptcies are not false positives.
LEAKAGE_REGEX = r"pymnt|(?<!pub_)rec_|recover|settle|hardship|out_prncp|^last_"

EMP_LENGTH_YEARS: dict[str, float] = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    **{f"{n} years": float(n) for n in range(2, 10)},
    "10+ years": 10.0,
}
HOME_OWNERSHIP_OTHER = {"OTHER", "NONE", "ANY"}


class LeakageError(ValueError):
    """A column that must never be a model input reached ``X``."""


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole calendar months from ``earlier`` to ``later`` (days ignored), as float64.

    Missing on either side gives NaN. Negative results are returned as is.
    """
    months = (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)
    return months.astype("float64")


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the cleaned frame with the derived and transformed columns."""
    out = df.copy()
    out["fico_mid"] = (out["fico_range_low"] + out["fico_range_high"]) / 2.0

    age = months_between(out["issue_d"], out["earliest_cr_line"])
    out["credit_age_months"] = age.where(age >= 0)

    emp = out["emp_length"]
    unknown = sorted(set(emp.dropna().unique()) - set(EMP_LENGTH_YEARS))
    if unknown:
        raise ValueError(f"unknown emp_length value(s): {unknown}")
    out["emp_length"] = emp.map(EMP_LENGTH_YEARS).astype("float64")

    home = out["home_ownership"]
    out["home_ownership"] = home.where(~home.isin(HOME_OWNERSHIP_OTHER), "OTHER")

    for col in ("annual_inc", "revol_bal"):
        out[col] = np.log1p(out[col].astype("float64"))
    return out


def check_model_columns(columns: Iterable[str], cfg: FeaturesConfig) -> None:
    """Raise ``LeakageError`` unless every column is a whitelisted, non-leaking feature."""
    whitelist = set(cfg.feature_names())
    restricted = {
        **dict.fromkeys(cfg.benchmark_only, "benchmark-only"),
        **dict.fromkeys(cfg.fairness_only, "fairness-only"),
        **dict.fromkeys(cfg.outcome_only, "outcome-only"),
        "y": "the target",
    }
    problems = []
    for col in columns:
        if col in restricted:
            problems.append(f"{col} ({restricted[col]})")
        elif re.search(LEAKAGE_REGEX, col):
            problems.append(f"{col} (matches leakage regex)")
        elif col not in whitelist:
            problems.append(f"{col} (not in the whitelist)")
    if problems:
        raise LeakageError("column(s) not allowed in X: " + ", ".join(problems))


def model_frame(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """``X``: exactly the whitelisted features, in config order, after the leakage check."""
    names = cfg.feature_names()
    check_model_columns(names, cfg)
    return df[names].copy()

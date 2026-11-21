"""Step 1.3 (D-04): the DuckDB vintage table agrees with the pandas profile."""

from __future__ import annotations

import pandas as pd
import pytest
from test_target import _profile_input

from credit_risk.data.sql import SQL_DIR, run_sql
from credit_risk.data.target import vintage_profile


def test_sql_file_exists() -> None:
    assert (SQL_DIR / "vintage_bad_rate.sql").is_file()


def test_vintage_bad_rate_sql_matches_pandas_profile() -> None:
    targeted = _profile_input()
    sql = run_sql("vintage_bad_rate", loans=targeted)
    prof = vintage_profile(targeted, ["dti"], flag_share=0.01)
    cols = ["quarter", "rows", "n_bad", "n_good", "n_indeterminate"]
    pd.testing.assert_frame_equal(
        sql[cols].astype({c: "int64" for c in cols[1:]}).astype({"quarter": object}),
        prof[cols].astype({c: "int64" for c in cols[1:]}).astype({"quarter": object}),
        check_dtype=False,
    )
    assert sql["bad_rate"].tolist() == pytest.approx(prof["bad_rate"].tolist())


def test_unknown_sql_name_raises() -> None:
    with pytest.raises(FileNotFoundError):
        run_sql("does_not_exist", loans=pd.DataFrame())

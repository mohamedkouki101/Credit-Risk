"""Step 1.1 (D-02): ``load_raw`` reads only allowed columns, each with an explicit dtype."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.config import REPO_ROOT, load_all_configs, load_base_config
from credit_risk.data.load import RAW_DTYPES, load_raw, raw_columns

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lc_accepted_columns.txt"
ALL_COLUMNS = [
    line.strip()
    for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]
NUMERIC = {c for c, t in RAW_DTYPES.items() if t == "float64"}

ROW = {
    "id": "68407277",
    "loan_amnt": "3600.0",
    "term": " 36 months",
    "int_rate": "13.99",
    "sub_grade": "C4",
    "emp_length": "10+ years",
    "annual_inc": "55000.0",
    "issue_d": "Dec-2015",
    "loan_status": "Fully Paid",
    "zip_code": "190xx",
    "earliest_cr_line": "Aug-2003",
    "revol_util": "29.7",
    "total_pymnt": "4421.7",  # post-origination: must never be read
    "last_fico_range_high": "564",  # post-origination: must never be read
}


def _write_csv_gz(path: Path, rows: list[dict[str, str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_raw_columns_are_keys_feature_sources_and_audit_groups() -> None:
    cfg = load_all_configs().features
    expected = (
        set(cfg.key_columns)
        | cfg.feature_source_columns()
        | set(cfg.benchmark_only)
        | set(cfg.fairness_only)
        | set(cfg.outcome_only)
    )
    cols = raw_columns(cfg)
    assert set(cols) == expected
    assert len(cols) == len(set(cols))


def test_every_raw_column_has_an_explicit_dtype() -> None:
    cols = raw_columns(load_all_configs().features)
    assert set(cols) <= set(RAW_DTYPES)
    assert set(RAW_DTYPES.values()) == {"float64", "str"}


def test_no_post_origination_column_is_read() -> None:
    cols = raw_columns(load_all_configs().features)
    assert not {"total_pymnt", "last_fico_range_high", "out_prncp", "recoveries_x"} & set(cols)


def test_load_raw_reads_only_allowed_columns_with_explicit_dtypes(tmp_path: Path) -> None:
    path = _write_csv_gz(tmp_path / "raw.csv.gz", [ROW, {"id": "Total amount funded: 1"}])
    df = load_raw(path)
    assert list(df.columns) == raw_columns(load_all_configs().features)
    assert "total_pymnt" not in df.columns
    assert len(df) == 2
    for col in df.columns:
        if col in NUMERIC:
            assert df[col].dtype == "float64", col
        else:
            assert pd.api.types.is_string_dtype(df[col]), col
    assert df.loc[0, "id"] == "68407277"
    assert df.loc[0, "term"] == " 36 months"  # whitespace is stripped in cleaning, not here
    assert df.loc[0, "fico_range_low"] != df.loc[0, "fico_range_low"]  # blank -> NaN


def test_load_raw_fails_loudly_on_text_in_a_numeric_column(tmp_path: Path) -> None:
    path = _write_csv_gz(tmp_path / "raw.csv.gz", [{**ROW, "revol_util": "29.7%"}])
    with pytest.raises(ValueError):
        load_raw(path)


@pytest.mark.slow
def test_load_raw_on_the_real_file_head() -> None:
    base = load_base_config()
    path = REPO_ROOT / base.paths.data_raw / base.data.kaggle.file
    if not path.exists():
        pytest.skip("raw file not downloaded")
    df = load_raw(path, nrows=5_000)
    assert len(df) == 5_000
    assert df["loan_amnt"].dtype == "float64"
    assert df["issue_d"].str.match(r"^[A-Z][a-z]{2}-\d{4}$").all()

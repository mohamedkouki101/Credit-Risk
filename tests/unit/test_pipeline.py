"""Step 1.3 (D-04): the ``data`` stage, end to end on a tiny synthetic raw file."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from test_load import ROW, _write_csv_gz

from credit_risk.__main__ import STAGES
from credit_risk.data.pipeline import run_data_stage
from credit_risk.data.split import SPLITS_END, SPLITS_START


def _row(**kw: str) -> dict[str, str]:
    base = {
        **ROW,
        "application_type": "Individual",
        "purpose": "credit_card",
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "fico_range_low": "700",
        "fico_range_high": "704",
        "dti": "12.5",
        "revol_bal": "1000",
        "issue_d": "Mar-2012",
    }
    return {**base, **kw}


@pytest.fixture
def raw_file(tmp_path: Path) -> Path:
    rows = [
        _row(id="1", loan_status="Fully Paid"),
        _row(id="2", loan_status="Charged Off", issue_d="Apr-2012"),
        _row(id="3", loan_status="Current", issue_d="May-2012"),
        _row(id="4", loan_status="Charged Off", term=" 60 months"),
        _row(id="4", loan_status="Charged Off", term=" 60 months"),  # duplicate id
        _row(id="5", loan_status="Current", issue_d="Jan-2017"),  # monitoring vintage
        _row(id="6", loan_status="Fully Paid", dti="-1"),  # dti -> missing + flag
        {"id": "Total amount funded in policy code 1: 123"},  # footer
    ]
    return _write_csv_gz(tmp_path / "raw.csv.gz", rows)


def test_data_stage_writes_clean_table_profile_and_logs(raw_file: Path, tmp_path: Path) -> None:
    out = run_data_stage(
        raw_path=raw_file,
        interim_dir=tmp_path / "interim",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        research_dir=tmp_path / "research",
    )
    clean = pd.read_parquet(out.clean_parquet)
    assert clean["id"].tolist() == ["1", "2", "3", "4", "5", "6"]
    assert clean["y"].tolist()[:2] == [0, 1]
    assert clean["y"].isna().tolist() == [False, False, True, True, True, False]

    meta = json.loads(out.meta_json.read_text(encoding="utf-8"))
    assert meta["raw_rows"] == 8
    assert meta["clean_rows"] == 6
    assert meta["drops"] == {
        "footer_empty_loan_amnt": 1,
        "issue_d_unparsed": 0,
        "earliest_cr_line_unparsed": 0,
        "duplicate_id": 1,
    }
    assert len(meta["schema_sha256"]) == 64

    profile = pd.read_csv(out.profile_csv)
    assert profile["quarter"].tolist() == ["2012Q1", "2012Q2"]
    assert profile["rows"].tolist() == [2, 2]

    log = pd.read_csv(out.cleaning_log_csv)
    assert log.set_index("rule").loc["dti_out_of_range", "rows"] == 1

    counts = pd.read_csv(out.population_csv).set_index("group")["rows"].to_dict()
    assert counts == {
        "modelled": 3,
        "application_type": 0,
        "indeterminate": 1,
        "term": 1,
        "issue_d_window": 1,
    }

    md = out.profile_md.read_text(encoding="utf-8")
    assert "# Data profile" in md
    assert "duplicate_id" in md
    assert "sql/vintage_bad_rate.sql" in md
    assert SPLITS_START in md and SPLITS_END in md  # filled by the features stage

    audit = pd.read_csv(out.audit_csv)
    assert audit["feature"].tolist()[:3] == ["loan_amnt", "purpose", "term"]
    assert set(audit["decision"]) <= {"keep", "drop"}
    assert pd.read_csv(out.iv_by_year_csv)["year"].unique().tolist() == [2012]
    assert out.audit_md.read_text(encoding="utf-8").startswith("# Feature audit")


def test_data_and_features_stages_are_wired_into_the_cli() -> None:
    assert STAGES["data"].__name__ == "run_data_cli"
    assert STAGES["features"].__name__ == "run_features_cli"

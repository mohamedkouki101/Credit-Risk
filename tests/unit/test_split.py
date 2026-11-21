"""Step 1.6 (D-06): out-of-time split materialisation and T-02."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.config import SplitsConfig, load_all_configs
from credit_risk.data.split import (
    SPLITS_END,
    SPLITS_START,
    SplitError,
    check_splits,
    make_splits,
    render_split_section,
    run_features_stage,
    split_summary,
)

SPLITS: SplitsConfig = load_all_configs().splits


def _table() -> pd.DataFrame:
    """One row per interesting case; ``y`` is set only for modelled rows (as build_target does)."""
    rows = [
        # id, issue month, y, segment, application_type, term
        ("1", "2007-06", 0, "core", "Individual", "36 months"),
        ("2", "2013-12", 1, "core", "Individual", "36 months"),  # last train month
        ("3", "2014-01", 0, "core", "Individual", "36 months"),  # first valid month
        ("4", "2014-12", 1, "core", "Individual", "36 months"),
        ("5", "2015-01", 0, "core", "Individual", "36 months"),  # first OOT month
        ("6", "2015-12", 1, "core", "Individual", "36 months"),
        ("7", "2015-06", None, "core", "Individual", "36 months"),  # indeterminate
        ("8", "2013-05", None, "out_of_population", "Individual", "60 months"),
        ("9", "2016-01", None, "out_of_population", "Individual", "36 months"),  # monitor
        ("10", "2018-12", None, "out_of_population", "Individual", "60 months"),  # monitor
        ("11", "2017-03", None, "out_of_population", "Joint App", "36 months"),
        ("12", "2007-05", None, "out_of_population", "Individual", "36 months"),
    ]
    df = pd.DataFrame(rows, columns=["id", "issue_d", "y", "segment", "application_type", "term"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%Y-%m")
    df["y"] = pd.array(df["y"], dtype="Int8")
    df["loan_amnt"] = 1000.0
    return df


def _ids(df: pd.DataFrame, idx: pd.Index) -> list[str]:
    return df.loc[idx, "id"].tolist()


def test_make_splits_assigns_rows_by_issue_month() -> None:
    df = _table()
    s = make_splits(df, SPLITS)
    assert list(s) == ["train", "valid", "test", "monitor"]
    assert _ids(df, s["train"]) == ["1", "2"]
    assert _ids(df, s["valid"]) == ["3", "4"]
    assert _ids(df, s["test"]) == ["5", "6"]
    assert _ids(df, s["monitor"]) == ["9", "10"]  # all terms, Individual only


def test_labelled_splits_contain_only_modelled_rows() -> None:
    df = _table()
    s = make_splits(df, SPLITS)
    for name in ("train", "valid", "test"):
        assert df.loc[s[name], "y"].notna().all(), name


def test_t02_split_dates_are_strictly_ordered_and_ids_disjoint() -> None:
    df = _table()
    s = make_splits(df, SPLITS)
    check_splits(df, s)  # does not raise
    d = {k: df.loc[v, "issue_d"] for k, v in s.items()}
    assert d["train"].max() < d["valid"].min()
    assert d["valid"].max() < d["test"].min()
    assert d["test"].max() < d["monitor"].min()
    ids = [set(df.loc[v, "id"]) for v in s.values()]
    assert sum(len(i) for i in ids) == len(set().union(*ids))


def test_t02_overlapping_dates_raise() -> None:
    df = _table()
    s = make_splits(df, SPLITS)
    bad = {**s, "valid": s["valid"].append(s["train"][-1:])}
    with pytest.raises(SplitError, match="order"):
        check_splits(df, bad)


def test_t02_shared_id_raises() -> None:
    df = _table()
    df.loc[4, "id"] = "3"  # an OOT row reuses a validation id
    s = make_splits(df, SPLITS)
    with pytest.raises(SplitError, match="id"):
        check_splits(df, s)


def test_split_summary_counts_and_bad_rates() -> None:
    df = _table()
    summary = split_summary(df, make_splits(df, SPLITS)).set_index("split")
    assert summary.loc["train", "rows"] == 2
    assert summary.loc["train", "n_bad"] == 1
    assert summary.loc["train", "bad_rate"] == pytest.approx(0.5)
    assert summary.loc["monitor", "rows"] == 2
    assert pd.isna(summary.loc["monitor", "bad_rate"])  # unlabelled


def test_render_split_section_has_markers_and_rows() -> None:
    df = _table()
    md = render_split_section(split_summary(df, make_splits(df, SPLITS)))
    assert md.startswith(SPLITS_START) and md.endswith(SPLITS_END)
    assert "| train |" in md and "| monitor |" in md


def test_features_stage_writes_parquets_without_monitor_labels(tmp_path: Path) -> None:
    interim, processed, results, reports = (tmp_path / d for d in ("i", "p", "r", "rep"))
    interim.mkdir()
    reports.mkdir()
    _table().to_parquet(interim / "loans_clean.parquet", index=False)
    (reports / "data_profile.md").write_text(
        f"# Data profile\n\n{SPLITS_START}\nplaceholder\n{SPLITS_END}\n", encoding="utf-8"
    )
    out = run_features_stage(
        interim_dir=interim, processed_dir=processed, results_dir=results, reports_dir=reports
    )
    frames = {k: pd.read_parquet(p) for k, p in out.split_parquets.items()}
    assert [len(f) for f in frames.values()] == [2, 2, 2, 2]
    assert "y" not in frames["monitor"].columns  # outcomes are not mature: no labels
    assert frames["test"]["y"].tolist() == [0, 1]
    assert pd.read_csv(out.summary_csv)["split"].tolist() == ["train", "valid", "test", "monitor"]
    profile = (reports / "data_profile.md").read_text(encoding="utf-8")
    assert "placeholder" not in profile and "| valid |" in profile

"""The ``data`` stage: download -> load -> clean -> target -> derive -> validate -> write.

Outputs (paths from ``configs/base.yaml``):

- ``data/interim/loans_clean.parquet``: every cleaned row with ``y``, ``segment``,
  ``exclusion_reason`` and the derived features (git-ignored).
- ``data/interim/loans_clean.meta.json``: raw file, row counts, drops, schema hash.
- ``results/vintage_profile.csv``, ``results/cleaning_log.csv``,
  ``results/population_counts.csv``: the numbers the docs quote.
- ``results/feature_audit.csv``, ``results/feature_iv_by_year.csv`` (D-05, train only).
- ``reports/data_profile.md`` and ``research/feature_audit.md``: rendered from those
  tables, never typed by hand.

The row-count identity ``clean = raw - logged drops`` is asserted, and the DuckDB vintage
table (``sql/vintage_bad_rate.sql``) must agree with the pandas profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import (
    REPO_ROOT,
    BaseConfig,
    ProjectConfig,
    load_all_configs,
    load_base_config,
)
from credit_risk.data.audit import feature_audit, render_audit_md
from credit_risk.data.clean import CleaningLog, clean
from credit_risk.data.download import download
from credit_risk.data.features import derive
from credit_risk.data.load import load_raw
from credit_risk.data.schema import LoanSchema
from credit_risk.data.split import SPLITS_END, SPLITS_START
from credit_risk.data.sql import run_sql
from credit_risk.data.target import build_target, render_profile_md, vintage_profile

POPULATION_GROUPS = ("modelled", "application_type", "term", "issue_d_window", "indeterminate")


@dataclass(frozen=True)
class DataStageOutputs:
    clean_parquet: Path
    meta_json: Path
    profile_csv: Path
    cleaning_log_csv: Path
    population_csv: Path
    profile_md: Path
    audit_csv: Path
    iv_by_year_csv: Path
    audit_md: Path


def build_loan_table(raw: pd.DataFrame, cfg: ProjectConfig) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean, label and derive ``raw``; validate it against ``LoanSchema``."""
    cleaned, log = clean(raw)
    table = derive(build_target(cleaned, cfg.target))
    LoanSchema.validate(table, lazy=True)  # report every failing check at once
    if len(table) != log.raw_rows - log.total_dropped:
        raise AssertionError("clean row count != raw rows - logged drops")
    return table, log


def population_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Rows per group: ``modelled`` (has ``y``) and each exclusion reason."""
    reasons = table["exclusion_reason"].fillna("modelled")
    counts = reasons.value_counts()
    return pd.DataFrame(
        {"group": POPULATION_GROUPS, "rows": [int(counts.get(g, 0)) for g in POPULATION_GROUPS]}
    )


def schema_sha256(table: pd.DataFrame) -> str:
    """Hash of column names and dtypes: changes whenever the table's layout changes."""
    layout = json.dumps([[c, str(t)] for c, t in table.dtypes.items()])
    return hashlib.sha256(layout.encode("utf-8")).hexdigest()


def _check_sql_agrees(table: pd.DataFrame, profile: pd.DataFrame) -> None:
    sql = run_sql("vintage_bad_rate", loans=table)
    cols = ["rows", "n_bad", "n_good", "n_indeterminate"]
    same = sql["quarter"].tolist() == profile["quarter"].tolist() and np.array_equal(
        sql[cols].to_numpy(dtype="int64"), profile[cols].to_numpy(dtype="int64")
    )
    if not same:
        raise AssertionError("sql/vintage_bad_rate.sql disagrees with vintage_profile")


def _profile_markdown(
    profile: pd.DataFrame, log: CleaningLog, counts: pd.DataFrame, flag_share: float
) -> str:
    lines = [render_profile_md(profile, flag_share).rstrip("\n"), "", "## Cleaning log", ""]
    lines += [f"Raw rows: {log.raw_rows:,}. Clean rows: {log.clean_rows:,}.", ""]
    lines += ["| Rule | Kind | Rows |", "|---|---|---:|"]
    lines += [f"| {r.rule} | {r.kind} | {r.rows:,} |" for r in log.to_frame().itertuples()]
    lines += ["", "## Population", "", "| Group | Rows |", "|---|---:|"]
    lines += [f"| {r.group} | {r.rows:,} |" for r in counts.itertuples()]
    lines += [
        "",
        "`modelled` rows have a label (`y`); every other group is excluded for the reason named.",
        "",
        "Cross-check: `sql/vintage_bad_rate.sql` (DuckDB) gives the same vintage counts.",
        "",
        SPLITS_START,
        "## Splits",
        "",
        "Not cut yet: run `uv run python -m credit_risk features`.",
        SPLITS_END,
        "",
    ]
    return "\n".join(lines)


def run_data_stage(
    cfg: ProjectConfig | None = None,
    base: BaseConfig | None = None,
    raw_path: Path | None = None,
    interim_dir: Path | None = None,
    results_dir: Path | None = None,
    reports_dir: Path | None = None,
    research_dir: Path | None = None,
) -> DataStageOutputs:
    """Run the whole data stage; defaults come from ``configs/``."""
    cfg = cfg or load_all_configs()
    base = base or load_base_config()
    if raw_path is None:
        if base.data.source != "kaggle":
            raise NotImplementedError("only the Kaggle variant is built (D-01 chose Kaggle)")
        raw_path = download("kaggle", base.data)
    interim_dir = interim_dir or REPO_ROOT / base.paths.data_interim
    results_dir = results_dir or REPO_ROOT / base.paths.results
    reports_dir = reports_dir or REPO_ROOT / base.paths.reports
    research_dir = research_dir or REPO_ROOT / "research"
    for d in (interim_dir, results_dir, reports_dir, research_dir):
        d.mkdir(parents=True, exist_ok=True)

    table, log = build_loan_table(load_raw(raw_path, cfg.features), cfg)
    flag_share = cfg.target.indeterminate_flag_share
    profile = vintage_profile(table, cfg.features.feature_names(), flag_share)
    _check_sql_agrees(table, profile)
    counts = population_counts(table)
    binning = cfg.scorecard.binning
    audit, iv_by_year = feature_audit(
        table, cfg.features.features, cfg.splits.train, binning.iv_min, binning.iv_leakage_flag
    )

    out = DataStageOutputs(
        clean_parquet=interim_dir / "loans_clean.parquet",
        meta_json=interim_dir / "loans_clean.meta.json",
        profile_csv=results_dir / "vintage_profile.csv",
        cleaning_log_csv=results_dir / "cleaning_log.csv",
        population_csv=results_dir / "population_counts.csv",
        profile_md=reports_dir / "data_profile.md",
        audit_csv=results_dir / "feature_audit.csv",
        iv_by_year_csv=results_dir / "feature_iv_by_year.csv",
        audit_md=research_dir / "feature_audit.md",
    )
    table.to_parquet(out.clean_parquet, index=False)
    meta = {
        "raw_file": raw_path.name,
        "raw_rows": log.raw_rows,
        "clean_rows": log.clean_rows,
        "drops": log.drops,
        "set_missing": log.flags,
        "schema_sha256": schema_sha256(table),
    }
    out.meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    profile.to_csv(out.profile_csv, index=False, lineterminator="\n")
    log.to_frame().to_csv(out.cleaning_log_csv, index=False, lineterminator="\n")
    counts.to_csv(out.population_csv, index=False, lineterminator="\n")
    out.profile_md.write_text(
        _profile_markdown(profile, log, counts, flag_share), encoding="utf-8", newline="\n"
    )
    audit.to_csv(out.audit_csv, index=False, lineterminator="\n")
    iv_by_year.to_csv(out.iv_by_year_csv, index=False, lineterminator="\n")
    out.audit_md.write_text(render_audit_md(audit, iv_by_year), encoding="utf-8", newline="\n")
    return out

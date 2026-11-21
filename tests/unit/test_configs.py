"""Step 0.5: every file in configs/ validates, and cross-file rules from the plan hold.

These guard the hard rules before any data code exists: feature groups never overlap
(leakage, data plan 5.2), every loan status has exactly one class (T-06 precondition),
splits are out-of-time and contiguous (5.5) and backtests never reach the OOT year.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from credit_risk.config import (
    BusinessConfig,
    FeaturesConfig,
    ProjectConfig,
    SplitsConfig,
    TargetConfig,
    load_all_configs,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lc_accepted_columns.txt"


@pytest.fixture(scope="module")
def cfg() -> ProjectConfig:
    return load_all_configs()


@pytest.fixture(scope="module")
def raw_columns() -> set[str]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _revalidate(model: Any, mutate: Any) -> None:
    data = model.model_dump(mode="json")
    mutate(data)
    type(model).model_validate(data)


# --- everything loads ------------------------------------------------------------------


def test_all_repo_configs_validate(cfg: ProjectConfig) -> None:
    assert cfg.base.global_seed == 42
    assert cfg.base.data.source == "kaggle"


def test_kaggle_hash_matches_data_decision(cfg: ProjectConfig) -> None:
    assert cfg.base.data.kaggle.sha256 == (
        "55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d"
    )


# --- target.yaml -----------------------------------------------------------------------


def test_status_map_covers_every_status_in_the_file(cfg: ProjectConfig) -> None:
    observed = {  # distinct non-null loan_status values in the v3 file (step 0.5)
        "Fully Paid",
        "Current",
        "Charged Off",
        "Late (31-120 days)",
        "In Grace Period",
        "Late (16-30 days)",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
        "Default",
    }
    assert set(cfg.target.status_to_class()) == observed


def test_status_in_two_classes_is_rejected(cfg: ProjectConfig) -> None:
    with pytest.raises(ValidationError, match="more than one class"):
        _revalidate(cfg.target, lambda d: d["status_map"]["good"].append("Charged Off"))


def test_population_window_must_be_ordered(cfg: ProjectConfig) -> None:
    def flip(d: dict[str, Any]) -> None:
        d["population"]["issue_d_start"], d["population"]["issue_d_end"] = "2016-01", "2007-06"

    with pytest.raises(ValidationError):
        _revalidate(cfg.target, flip)


def test_bad_month_format_is_rejected(cfg: ProjectConfig) -> None:
    with pytest.raises(ValidationError):
        _revalidate(cfg.target, lambda d: d["population"].update(issue_d_start="2007-6"))


# --- splits.yaml -----------------------------------------------------------------------


def test_splits_match_preregistered_boundaries(cfg: ProjectConfig) -> None:
    s = cfg.splits
    assert (s.train.start, s.train.end) == ("2007-06", "2013-12")
    assert (s.valid.start, s.valid.end) == ("2014-01", "2014-12")
    assert (s.test.start, s.test.end) == ("2015-01", "2015-12")
    assert (s.monitor.start, s.monitor.end) == ("2016-01", "2018-12")


def test_overlapping_splits_are_rejected(cfg: ProjectConfig) -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        _revalidate(cfg.splits, lambda d: d["valid"].update(start="2013-06"))


def test_gap_between_splits_is_rejected(cfg: ProjectConfig) -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        _revalidate(cfg.splits, lambda d: d["test"].update(start="2015-03"))


def test_backtest_fold_reaching_oot_year_is_rejected(cfg: ProjectConfig) -> None:
    def leak(d: dict[str, Any]) -> None:
        d["backtest_folds"][-1]["test"] = {"start": "2015-01", "end": "2015-12"}

    with pytest.raises(ValidationError, match="OOT"):
        _revalidate(cfg.splits, leak)


def test_labelled_splits_lie_inside_target_population(cfg: ProjectConfig) -> None:
    pop = cfg.target.population
    assert cfg.splits.train.start == pop.issue_d_start
    assert cfg.splits.test.end == pop.issue_d_end


# --- features.yaml ---------------------------------------------------------------------


def test_benchmark_column_in_whitelist_is_rejected(cfg: ProjectConfig) -> None:
    def leak(d: dict[str, Any]) -> None:
        d["features"].append({"name": "sub_grade", "kind": "ordinal"})

    with pytest.raises(ValidationError, match="sub_grade"):
        _revalidate(cfg.features, leak)


def test_outcome_column_as_feature_source_is_rejected(cfg: ProjectConfig) -> None:
    def leak(d: dict[str, Any]) -> None:
        d["features"].append({"name": "paid", "kind": "numeric", "sources": ["total_rec_prncp"]})

    with pytest.raises(ValidationError, match="total_rec_prncp"):
        _revalidate(cfg.features, leak)


def test_excluded_pattern_as_feature_is_rejected(cfg: ProjectConfig) -> None:
    def leak(d: dict[str, Any]) -> None:
        d["features"].append({"name": "last_fico_range_high", "kind": "numeric"})

    with pytest.raises(ValidationError, match="excluded"):
        _revalidate(cfg.features, leak)


def test_incumbent_and_fairness_columns_are_never_features(cfg: ProjectConfig) -> None:
    sources = cfg.features.feature_source_columns()
    for col in ("grade", "sub_grade", "int_rate", "installment", "addr_state", "zip_code"):
        assert col not in sources


def test_every_config_column_exists_in_raw_file(cfg: ProjectConfig, raw_columns: set[str]) -> None:
    f = cfg.features
    named = (
        f.feature_source_columns()
        | set(f.key_columns)
        | set(f.benchmark_only)
        | set(f.fairness_only)
        | set(f.outcome_only)
    )
    assert named - raw_columns == set()


def test_every_exclusion_pattern_matches_a_real_column(
    cfg: ProjectConfig, raw_columns: set[str]
) -> None:
    for rule in cfg.features.excluded:
        assert any(fnmatch(c, rule.pattern) for c in raw_columns), rule.pattern


# --- monotone.yaml, scorecard.yaml, reason_text.yaml, business.yaml --------------------


def test_monotone_directions_follow_roadmap(cfg: ProjectConfig) -> None:
    c = cfg.monotone.constraints
    assert c["fico_mid"] == -1 and c["annual_inc"] == -1 and c["credit_age_months"] == -1
    assert c["dti"] == 1 and c["revol_util"] == 1 and c["inq_last_6mths"] == 1


def test_monotone_vector_has_one_entry_per_feature(cfg: ProjectConfig) -> None:
    names = cfg.features.feature_names()
    vector = cfg.monotone.vector(names)
    assert len(vector) == len(names)
    assert vector[names.index("fico_mid")] == -1
    assert vector[names.index("loan_amnt")] == 0


def test_monotone_on_unknown_feature_is_rejected(cfg: ProjectConfig) -> None:
    data = cfg.model_dump(mode="json")
    data["monotone"]["constraints"]["sub_grade"] = 1
    with pytest.raises(ValidationError, match="sub_grade"):
        ProjectConfig.model_validate(data)


def test_scorecard_scaling_is_600_at_50_to_1_pdo_20(cfg: ProjectConfig) -> None:
    sc = cfg.scorecard.scaling
    assert (sc.base_score, sc.base_odds, sc.pdo) == (600, 50, 20)


def test_every_feature_has_exactly_one_reason_family(cfg: ProjectConfig) -> None:
    assigned = [f for fam in cfg.reason_text.families.values() for f in fam.features]
    assert sorted(assigned) == sorted(cfg.features.feature_names())


def test_feature_without_reason_family_is_rejected(cfg: ProjectConfig) -> None:
    data = cfg.model_dump(mode="json")
    data["features"]["features"].append({"name": "open_il_12m", "kind": "numeric"})
    with pytest.raises(ValidationError, match="reason family"):
        ProjectConfig.model_validate(data)


def test_business_grids(cfg: ProjectConfig) -> None:
    assert cfg.business.margin_grid == [0.05, 0.10, 0.15]
    assert cfg.business.approval_rates == [0.6, 0.7, 0.8]


def test_approval_rate_outside_unit_interval_is_rejected(cfg: ProjectConfig) -> None:
    with pytest.raises(ValidationError):
        _revalidate(cfg.business, lambda d: d.update(approval_rates=[0.7, 1.2]))


def test_model_classes_are_exported() -> None:
    for model in (TargetConfig, SplitsConfig, FeaturesConfig, BusinessConfig):
        assert model.model_config.get("extra") == "forbid"

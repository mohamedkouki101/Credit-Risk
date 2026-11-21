"""Pydantic models that load and validate the YAML files in ``configs/``.

Every config model forbids unknown keys, so a typo in a YAML file fails loudly
instead of being silently ignored. Paths in configs are relative to the repo root.
Months are ``"YYYY-MM"`` strings (inclusive bounds); they compare correctly as strings.

``load_all_configs()`` returns a ``ProjectConfig`` that also checks the rules that span
files: monotone constraints and reason families refer only to whitelisted features, and
the splits lie inside the target population window.
"""

from __future__ import annotations

from fnmatch import fnmatch
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

YearMonth = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
UnitInterval = Annotated[float, Field(gt=0.0, le=1.0)]
TargetClass = Literal["bad", "good", "indeterminate"]


def next_month(month: str) -> str:
    """``"2013-12"`` -> ``"2014-01"``."""
    year, mon = int(month[:4]), int(month[5:7])
    return f"{year + mon // 12:04d}-{mon % 12 + 1:02d}"


class StrictModel(BaseModel):
    """Base for all config models: unknown keys are an error, instances are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MonthRange(StrictModel):
    """Inclusive range of issue months."""

    start: YearMonth
    end: YearMonth

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


# --- base.yaml --------------------------------------------------------------------------


class PathsConfig(StrictModel):
    """Project directories, relative to the repo root."""

    data_raw: Path
    data_interim: Path
    data_processed: Path
    models: Path
    results: Path
    reports: Path


class KaggleSource(StrictModel):
    dataset: str
    version: int = Field(ge=1)
    file: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ZenodoSource(StrictModel):
    record: int
    file: str
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class DataConfig(StrictModel):
    """Which source is primary (decided in D-01) and the hashes downloads must match."""

    source: Literal["kaggle", "zenodo"]
    kaggle: KaggleSource
    zenodo: ZenodoSource


class BaseConfig(StrictModel):
    """``configs/base.yaml``: paths, the single global seed and the data source."""

    global_seed: int = Field(ge=0)
    paths: PathsConfig
    data: DataConfig


# --- target.yaml ------------------------------------------------------------------------


class PopulationConfig(StrictModel):
    application_type: str
    term: str
    issue_d_start: YearMonth
    issue_d_end: YearMonth

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.issue_d_start > self.issue_d_end:
            raise ValueError("issue_d_start is after issue_d_end")
        return self


class StatusMap(StrictModel):
    bad: list[str] = Field(min_length=1)
    good: list[str] = Field(min_length=1)
    indeterminate: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        seen: dict[str, str] = {}
        for cls in ("bad", "good", "indeterminate"):
            for status in getattr(self, cls):
                if status in seen:
                    raise ValueError(
                        f"loan_status {status!r} is in more than one class ({seen[status]}, {cls})"
                    )
                seen[status] = cls
        return self


class TargetConfig(StrictModel):
    """``configs/target.yaml``: population and loan_status -> y mapping (data plan 5.3a).

    ``y = 1`` is bad (charge-off/default), ``y = 0`` good; indeterminate rows are excluded.
    """

    population: PopulationConfig
    status_map: StatusMap
    policy_flag_prefix: str
    indeterminate_flag_share: UnitInterval

    def status_to_class(self) -> dict[str, TargetClass]:
        sm = self.status_map
        out: dict[str, TargetClass] = {}
        out.update(dict.fromkeys(sm.bad, "bad"))
        out.update(dict.fromkeys(sm.good, "good"))
        out.update(dict.fromkeys(sm.indeterminate, "indeterminate"))
        return out


# --- splits.yaml ------------------------------------------------------------------------


class BacktestFold(StrictModel):
    train_end: YearMonth
    test: MonthRange


class SplitsConfig(StrictModel):
    """``configs/splits.yaml``: out-of-time splits on ``date_column`` (data plan 5.5)."""

    date_column: str
    train: MonthRange
    valid: MonthRange
    test: MonthRange
    monitor: MonthRange
    backtest_folds: list[BacktestFold] = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous_and_no_oot_leak(self) -> Self:
        order = [
            ("train", self.train),
            ("valid", self.valid),
            ("test", self.test),
            ("monitor", self.monitor),
        ]
        for (name_a, a), (name_b, b) in pairwise(order):
            if next_month(a.end) != b.start:
                raise ValueError(
                    f"splits must be contiguous and ordered: {name_a} ends {a.end}, "
                    f"{name_b} starts {b.start}"
                )
        for i, fold in enumerate(self.backtest_folds):
            if fold.test.end >= self.test.start:
                raise ValueError(f"backtest fold {i} reaches the OOT test window")
            if next_month(fold.train_end) != fold.test.start:
                raise ValueError(f"backtest fold {i}: test must start right after train_end")
            if fold.train_end < self.train.start:
                raise ValueError(f"backtest fold {i}: train_end before the train start")
        return self


# --- features.yaml ----------------------------------------------------------------------


class FeatureSpec(StrictModel):
    """One model input. ``sources`` are the raw columns it is derived from (default: itself)."""

    name: str
    kind: Literal["numeric", "categorical", "ordinal"]
    sources: list[str] | None = None
    transform: str | None = None

    def source_columns(self) -> list[str]:
        return self.sources if self.sources is not None else [self.name]


class ExclusionRule(StrictModel):
    pattern: str
    reason: str


class FeaturesConfig(StrictModel):
    """``configs/features.yaml``: the application-time whitelist (data plan 5.2).

    Anything not whitelisted is excluded by default. ``benchmark_only``,
    ``fairness_only`` and ``outcome_only`` columns may be loaded but never reach ``X``.
    ``key_columns`` (id, issue date) may feed a derivation but are never features.
    """

    key_columns: list[str]
    features: list[FeatureSpec] = Field(min_length=1)
    benchmark_only: list[str]
    fairness_only: list[str]
    outcome_only: list[str]
    excluded: list[ExclusionRule]

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature_source_columns(self) -> set[str]:
        return {c for f in self.features for c in f.source_columns()}

    @model_validator(mode="after")
    def _no_leakage(self) -> Self:
        names = self.feature_names()
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate feature names: {dupes}")

        groups = {
            "benchmark_only": set(self.benchmark_only),
            "fairness_only": set(self.fairness_only),
            "outcome_only": set(self.outcome_only),
        }
        group_items = list(groups.items())
        for i, (name_a, a) in enumerate(group_items):
            for name_b, b in group_items[i + 1 :]:
                if a & b:
                    raise ValueError(f"{name_a} and {name_b} overlap: {sorted(a & b)}")
        restricted = set().union(*groups.values())

        for f in self.features:
            used = {f.name, *f.source_columns()}
            if hit := sorted(used & restricted):
                raise ValueError(
                    f"feature {f.name!r} uses restricted column(s) {hit} "
                    "(benchmark, fairness or outcome only)"
                )
            if f.name in self.key_columns:
                raise ValueError(f"key column {f.name!r} cannot be a feature")
            for col in used:
                for rule in self.excluded:
                    if fnmatch(col, rule.pattern):
                        raise ValueError(
                            f"feature {f.name!r} uses {col!r}, excluded by {rule.pattern!r}: "
                            f"{rule.reason}"
                        )
        return self


# --- monotone.yaml, scorecard.yaml, reason_text.yaml, business.yaml ----------------------


class MonotoneConfig(StrictModel):
    """LightGBM directions on P(bad): +1 PD rises with the feature, -1 falls, unlisted 0."""

    constraints: dict[str, Literal[-1, 0, 1]]
    methods: list[Literal["basic", "intermediate", "advanced"]] = Field(min_length=1)

    def vector(self, feature_names: list[str]) -> list[int]:
        """``monotone_constraints`` in the column order of ``feature_names``."""
        return [self.constraints.get(name, 0) for name in feature_names]


class ScalingConfig(StrictModel):
    """PDO scaling. Odds are good:bad; ``base_score`` points at ``base_odds``:1."""

    base_score: float
    base_odds: float = Field(gt=0)
    pdo: float = Field(gt=0)


class BinningConfig(StrictModel):
    monotonic_trend: str
    min_bin_size: float = Field(gt=0, le=0.5)
    iv_min: float = Field(ge=0)
    iv_max: float = Field(gt=0)
    iv_leakage_flag: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.iv_min < self.iv_leakage_flag <= self.iv_max:
            raise ValueError("need iv_min < iv_leakage_flag <= iv_max")
        return self


class ScorecardConfig(StrictModel):
    scaling: ScalingConfig
    binning: BinningConfig
    n_score_bands: int = Field(ge=2)


class ReasonFamily(StrictModel):
    features: list[str] = Field(min_length=1)
    text: str = Field(min_length=1)


class ReasonTextConfig(StrictModel):
    """Feature families -> adverse-action reason text. ``placeholder`` until reviewed (X-02)."""

    placeholder: bool
    max_reasons: int = Field(ge=1, le=4)
    families: dict[str, ReasonFamily] = Field(min_length=1)


class LgdConfig(StrictModel):
    source: Literal["realised_train_chargeoffs"]
    sensitivity_pp: float = Field(ge=0, lt=1)


class BusinessConfig(StrictModel):
    """Business-view assumptions (EV-03, SA-05). Values are illustrative and Unverified."""

    lgd: LgdConfig
    ead_column: str
    margin_grid: list[UnitInterval] = Field(min_length=1)
    approval_rates: list[UnitInterval] = Field(min_length=1)


# --- all configs together ---------------------------------------------------------------

CONFIG_FILES: dict[str, str] = {
    "base": "base.yaml",
    "target": "target.yaml",
    "splits": "splits.yaml",
    "features": "features.yaml",
    "monotone": "monotone.yaml",
    "scorecard": "scorecard.yaml",
    "reason_text": "reason_text.yaml",
    "business": "business.yaml",
}


class ProjectConfig(StrictModel):
    """Every config file, plus the rules that span files."""

    base: BaseConfig
    target: TargetConfig
    splits: SplitsConfig
    features: FeaturesConfig
    monotone: MonotoneConfig
    scorecard: ScorecardConfig
    reason_text: ReasonTextConfig
    business: BusinessConfig

    @model_validator(mode="after")
    def _cross_file(self) -> Self:
        names = self.features.feature_names()
        if unknown := sorted(set(self.monotone.constraints) - set(names)):
            raise ValueError(f"monotone constraint on unknown feature(s) {unknown}")

        assigned = [f for fam in self.reason_text.families.values() for f in fam.features]
        if unknown := sorted(set(assigned) - set(names)):
            raise ValueError(f"reason family lists unknown feature(s) {unknown}")
        if twice := sorted({f for f in assigned if assigned.count(f) > 1}):
            raise ValueError(f"feature(s) {twice} are in more than one reason family")
        if missing := sorted(set(names) - set(assigned)):
            raise ValueError(f"feature(s) {missing} have no reason family")

        pop = self.target.population
        if self.splits.train.start < pop.issue_d_start or self.splits.test.end > pop.issue_d_end:
            raise ValueError("labelled splits must lie inside the target population window")
        if self.splits.date_column not in self.features.key_columns:
            raise ValueError(f"split date column {self.splits.date_column!r} is not a key column")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return raw


def load_config[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """Read a YAML file and validate it against ``model``.

    Raises ``ValueError`` if the file is not a YAML mapping and
    ``pydantic.ValidationError`` if it does not match the model.
    """
    return model.model_validate(_read_yaml(path))


def load_base_config(config_dir: Path = CONFIG_DIR) -> BaseConfig:
    """Load ``base.yaml`` from ``config_dir``."""
    return load_config(config_dir / "base.yaml", BaseConfig)


def load_all_configs(config_dir: Path = CONFIG_DIR) -> ProjectConfig:
    """Load and cross-validate every file in ``config_dir``."""
    raw = {key: _read_yaml(config_dir / fname) for key, fname in CONFIG_FILES.items()}
    return ProjectConfig.model_validate(raw)

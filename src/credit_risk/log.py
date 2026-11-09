"""Append-only experiment log: ``results/experiments.jsonl`` (schema in PLAN §7).

One JSON object per model run, validated by pydantic before it is written.
Probabilities (``bad_rate_eval``, metrics) refer to the bad class (``y = 1``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

# The OOT test set only ever appears as stage "oot" (written by `credit_risk final`).
STAGE_PATTERN = r"^(validation|oot|backtest_fold_\d+)$"


class Metrics(BaseModel):
    """Evaluation metrics. All optional: a run logs what it computed.

    ``auc``/``gini``/``ks`` are ranking metrics for P(bad); ``brier`` and ``log_loss``
    are on the probability scale; ``cal_slope``/``cal_intercept`` are on log-odds.
    """

    model_config = ConfigDict(extra="forbid")

    auc: float | None = None
    auc_ci: tuple[float, float] | None = None
    ks: float | None = None
    gini: float | None = None
    pr_auc: float | None = None
    brier: float | None = None
    log_loss: float | None = None
    cal_slope: float | None = None
    cal_intercept: float | None = None


class ExperimentRecord(BaseModel):
    """One model run. Field names follow PLAN §7 exactly."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    config_hash: Sha256
    data_hash: Sha256
    stage: str = Field(pattern=STAGE_PATTERN)
    model: str = Field(min_length=1)
    arm: str | None = None
    ablation: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(ge=0)
    n_train: int = Field(ge=0)
    n_eval: int = Field(ge=0)
    bad_rate_eval: Probability
    metrics: Metrics = Field(default_factory=Metrics)
    artefacts: list[str] = Field(default_factory=list)
    duration_s: float = Field(ge=0.0)
    notes: str = ""
    prereg_tag: str | None = None


def append_experiment(record: ExperimentRecord, path: Path) -> None:
    """Append ``record`` as one JSON line to ``path``, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def read_experiments(path: Path) -> list[ExperimentRecord]:
    """Read and validate every record in ``path``."""
    with path.open(encoding="utf-8") as f:
        return [ExperimentRecord.model_validate_json(line) for line in f if line.strip()]

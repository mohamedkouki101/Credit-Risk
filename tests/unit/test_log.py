"""Step 0.3 (E-02): experiment-log writer, schema from PLAN §7."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from credit_risk.log import ExperimentRecord, append_experiment, read_experiments

SHA = "sha256:" + "a" * 64


def _record(**overrides: Any) -> ExperimentRecord:
    fields: dict[str, Any] = {
        "run_id": "2026-11-08T12:00:00Z_m0_baseline",
        "git_sha": "abc1234",
        "config_hash": SHA,
        "data_hash": SHA,
        "stage": "validation",
        "model": "M0",
        "params": {},
        "seed": 42,
        "n_train": 1000,
        "n_eval": 200,
        "bad_rate_eval": 0.18,
        "metrics": {"auc": 0.5, "auc_ci": [0.48, 0.52], "brier": 0.15},
        "duration_s": 0.1,
    }
    fields.update(overrides)
    return ExperimentRecord.model_validate(fields)


def test_append_writes_one_json_line_per_run(tmp_path: Path) -> None:
    log_path = tmp_path / "results" / "experiments.jsonl"
    append_experiment(_record(), log_path)
    append_experiment(_record(run_id="second", model="M1"), log_path)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["model"] == "M1"


def test_read_round_trips_records(tmp_path: Path) -> None:
    log_path = tmp_path / "experiments.jsonl"
    rec = _record()
    append_experiment(rec, log_path)
    assert read_experiments(log_path) == [rec]


@pytest.mark.parametrize("stage", ["validation", "oot", "backtest_fold_0", "backtest_fold_12"])
def test_valid_stages_accepted(stage: str) -> None:
    assert _record(stage=stage).stage == stage


@pytest.mark.parametrize("stage", ["test", "train", "backtest_fold_k", "OOT"])
def test_invalid_stages_rejected(stage: str) -> None:
    with pytest.raises(ValidationError):
        _record(stage=stage)


def test_malformed_hash_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(config_hash="md5:abc")


def test_bad_rate_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        _record(bad_rate_eval=1.5)


def test_unknown_top_level_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(acc=0.9)

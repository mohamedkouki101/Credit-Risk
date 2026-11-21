"""Step 1.5 (P-01): generated pre-registration tables and the check_prereg rules."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from credit_risk.config import CONFIG_DIR, REPO_ROOT, MonthRange, load_all_configs
from credit_risk.reporting.prereg import (
    BLOCK_END,
    BLOCK_START,
    config_hashes,
    fill_block,
    split_counts,
)


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_prereg.py"
    spec = importlib.util.spec_from_file_location("check_prereg", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_prereg"] = module
    spec.loader.exec_module(module)
    return module


CP = _load_script()


# --- generated tables ---------------------------------------------------------------------


def test_split_counts_sum_quarters_inside_each_window() -> None:
    profile = pd.DataFrame(
        {
            "quarter": ["2013Q4", "2014Q1", "2014Q4", "2015Q1"],
            "n_bad": [1, 2, 3, 4],
            "n_good": [10, 20, 30, 40],
            "n_indeterminate": [0, 0, 1, 2],
        }
    )
    windows = {
        "train": MonthRange(start="2013-10", end="2013-12"),
        "valid": MonthRange(start="2014-01", end="2014-12"),
        "test": MonthRange(start="2015-01", end="2015-12"),
    }
    out = split_counts(profile, windows).set_index("split")
    assert out.loc["valid", "n_labelled"] == 55
    assert out.loc["valid", "n_bad"] == 5
    assert out.loc["test", "n_indeterminate"] == 2
    assert out.loc["train", "first_quarter"] == "2013Q4"
    assert "bad_rate" not in out.columns  # counts only: no outcome rates pre-registered


def test_config_hashes_ignore_line_endings(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_bytes(b"x: 1\ny: 2\n")
    lf = config_hashes(tmp_path)
    (tmp_path / "a.yaml").write_bytes(b"x: 1\r\ny: 2\r\n")
    assert config_hashes(tmp_path) == lf
    assert len(lf["a.yaml"]) == 64


def test_repo_config_hashes_cover_every_config() -> None:
    assert set(config_hashes(CONFIG_DIR)) == {p.name for p in CONFIG_DIR.glob("*.yaml")}


def test_fill_block_replaces_only_the_marked_block() -> None:
    doc = f"head\n{BLOCK_START}\nold\n{BLOCK_END}\ntail\n"
    out = fill_block(doc, "new table")
    assert out == f"head\n{BLOCK_START}\nnew table\n{BLOCK_END}\ntail\n"
    with pytest.raises(ValueError, match="marker"):
        fill_block("no markers here", "x")


def test_repo_prereg_has_the_criteria_and_the_generated_block() -> None:
    text = (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        assert f"| {cid} |" in text
    assert BLOCK_START in text and BLOCK_END in text
    splits = load_all_configs().splits
    assert splits.train.start in text and splits.test.end in text
    assert "## Amendments" in text


# --- check_prereg decision rules ------------------------------------------------------------


def _facts(**kw: object) -> object:
    base: dict[str, object] = {
        "tag_commit": None,
        "prereg_in_tag": False,
        "first_models_commit": None,
        "tag_precedes_models": False,
        "changed_frozen": [],
        "prereg_text": "",
    }
    base.update(kw)
    return CP.GitFacts(**base)


def test_no_tag_and_no_model_code_passes() -> None:
    assert CP.evaluate(_facts()) == []


def test_model_code_without_tag_fails() -> None:
    errors = CP.evaluate(_facts(first_models_commit="b" * 40))
    assert any("tag" in e for e in errors)


def test_tag_must_contain_prereg_file() -> None:
    errors = CP.evaluate(_facts(tag_commit="a" * 40, prereg_in_tag=False))
    assert any("PREREGISTRATION.md" in e for e in errors)


def test_tag_before_model_code_passes() -> None:
    facts = _facts(
        tag_commit="a" * 40,
        prereg_in_tag=True,
        first_models_commit="b" * 40,
        tag_precedes_models=True,
    )
    assert CP.evaluate(facts) == []


def test_model_code_before_tag_fails() -> None:
    facts = _facts(
        tag_commit="a" * 40,
        prereg_in_tag=True,
        first_models_commit="b" * 40,
        tag_precedes_models=False,
    )
    assert any("before" in e for e in CP.evaluate(facts))


def test_changed_frozen_config_needs_an_amendment() -> None:
    facts = _facts(tag_commit="a" * 40, prereg_in_tag=True, changed_frozen=["configs/splits.yaml"])
    assert any("configs/splits.yaml" in e for e in CP.evaluate(facts))
    amended = "## Amendments\n\n### 2027-01-01 · configs/splits.yaml\nWhy: ...\n"
    facts = _facts(
        tag_commit="a" * 40,
        prereg_in_tag=True,
        changed_frozen=["configs/splits.yaml"],
        prereg_text=amended,
    )
    assert CP.evaluate(facts) == []


@pytest.mark.skipif(
    not (REPO_ROOT / ".git").exists() or shutil.which("git") is None, reason="needs the git repo"
)
def test_check_prereg_runs_read_only_on_this_repo() -> None:
    facts = CP.collect_facts(REPO_ROOT)
    assert isinstance(CP.evaluate(facts), list)


def test_split_counts_include_a_quarter_the_window_starts_inside() -> None:
    """The train window starts 2007-06, inside 2007Q2: that quarter's loans are train loans."""
    profile = pd.DataFrame(
        {
            "quarter": ["2007Q2", "2007Q3"],
            "n_bad": [3, 4],
            "n_good": [21, 40],
            "n_indeterminate": [0, 0],
        }
    )
    out = split_counts(profile, {"train": MonthRange(start="2007-06", end="2007-12")})
    assert out.loc[0, "n_labelled"] == 68
    assert out.loc[0, "first_quarter"] == "2007Q2"


def test_registered_counts_match_the_materialised_splits() -> None:
    """The counts in PREREGISTRATION.md equal results/split_summary.csv (features stage)."""
    summary_path = REPO_ROOT / "results" / "split_summary.csv"
    if not summary_path.exists():
        pytest.skip("features stage not run")
    text = (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    for r in pd.read_csv(summary_path).itertuples(index=False):
        if r.split != "monitor":
            assert f"| {r.rows:,} | {int(r.n_bad):,} |" in text, r.split

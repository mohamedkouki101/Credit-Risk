"""Step 0.3 (E-02): CI workflow, pre-commit config and Makefile are well formed."""

import re
from pathlib import Path
from typing import Any

import yaml

from credit_risk.__main__ import STAGES

ROOT = Path(__file__).resolve().parents[2]


def _yaml(rel: str) -> Any:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def _run_commands(workflow: dict[str, Any]) -> list[str]:
    return [
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    ]


def test_ci_runs_sync_lint_typecheck_fast_tests_in_order() -> None:
    commands = " \n".join(_run_commands(_yaml(".github/workflows/ci.yml")))
    expected = ["uv sync", "ruff check", "mypy src", 'pytest -m "not slow"']
    positions = [commands.find(c) for c in expected]
    assert all(p >= 0 for p in positions), f"missing step in CI: {commands}"
    assert positions == sorted(positions), "CI steps out of order"


def test_slow_workflow_is_manual_only() -> None:
    triggers = _yaml(".github/workflows/slow.yml")[True]  # PyYAML reads `on:` as True
    assert set(triggers) == {"workflow_dispatch"}


def test_precommit_blocks_files_over_5mb_and_runs_ruff() -> None:
    hooks = {
        h["id"]: h for repo in _yaml(".pre-commit-config.yaml")["repos"] for h in repo["hooks"]
    }
    assert {"ruff-check", "ruff-format", "end-of-file-fixer", "check-added-large-files"} <= set(
        hooks
    )
    assert "--maxkb=5120" in hooks["check-added-large-files"]["args"]


def test_makefile_wraps_every_cli_stage() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z]+):", makefile, flags=re.MULTILINE))
    assert set(STAGES) | {"setup", "test", "lint", "api", "app"} <= targets
    for stage in STAGES:
        assert f"python -m credit_risk {stage}" in makefile

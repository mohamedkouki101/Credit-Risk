"""Step 0.2 (E-01): the environment smoke script runs end to end.

Acceptance from roadmap E-01: ``uv run python scripts/smoke_env.py`` exits 0.
Slow because it fits OptBinning, LightGBM and SHAP on synthetic data.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_env.py"


@pytest.mark.slow
def test_smoke_env_script_exits_zero_and_times_three_steps() -> None:
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for step in ("optbinning_scorecard", "lightgbm_monotone", "shap_tree_explainer"):
        assert f"[timing] {step}" in result.stdout

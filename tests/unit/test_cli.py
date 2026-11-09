"""Step 0.3 (E-02): ``python -m credit_risk <stage>`` CLI."""

import subprocess
import sys

import numpy as np
import pytest

from credit_risk.__main__ import STAGES, main
from credit_risk.seed import set_all_seeds

EXPECTED_STAGES = (
    "data",
    "features",
    "train",
    "evaluate",
    "explain",
    "fairness",
    "monitor",
    "report",
    "final",
    "export",
)


def test_cli_exposes_planned_stages() -> None:
    assert tuple(STAGES) == EXPECTED_STAGES


@pytest.mark.parametrize("stage", EXPECTED_STAGES)
def test_unimplemented_stage_fails_loudly(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([stage]) == 1
    assert f"stage '{stage}' is not implemented yet" in capsys.readouterr().err


def test_unknown_stage_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["fit_everything"])
    assert exc.value.code == 2


def test_cli_seeds_from_base_config() -> None:
    # Checks the legacy global RNG on purpose: that is what the CLI must seed.
    main(["data"])
    after_cli = np.random.rand(3)  # noqa: NPY002
    set_all_seeds(42)
    assert np.array_equal(after_cli, np.random.rand(3))  # noqa: NPY002


def test_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "credit_risk", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "evaluate" in result.stdout

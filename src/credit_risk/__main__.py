"""Pipeline CLI: ``uv run python -m credit_risk <stage>``.

Works the same on Windows and Linux; the Makefile only wraps these commands.
Every invocation seeds all RNGs from ``configs/base.yaml`` before running a stage.
Stages not yet built exit with code 1 so a pipeline never silently "succeeds".
``final`` is the only stage allowed to touch the OOT test set.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from credit_risk.config import load_base_config
from credit_risk.seed import set_all_seeds

StageFn = Callable[[], int]

STAGE_HELP: dict[str, str] = {
    "data": "download, build target, clean (D-02..D-04)",
    "features": "derive features, splits (D-04, D-06)",
    "train": "fit baselines, scorecard, LightGBM (M-01..M-09)",
    "evaluate": "validation metrics with CIs (EV-01, EV-02)",
    "explain": "SHAP and reason codes (X-01..X-05)",
    "fairness": "fairness audit (F-01..F-04)",
    "monitor": "PSI monitoring on 2016-2018 (MR-02)",
    "report": "render README numbers, model card, validation report",
    "final": "ONE-TIME out-of-time test run; writes results/final.lock",
    "export": "export artefacts and manifest.json",
}


def _not_implemented(stage: str) -> StageFn:
    def run() -> int:
        print(f"stage '{stage}' is not implemented yet", file=sys.stderr)
        return 1

    return run


STAGES: dict[str, StageFn] = {name: _not_implemented(name) for name in STAGE_HELP}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit_risk", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True, metavar="<stage>")
    for name, help_text in STAGE_HELP.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_all_seeds(load_base_config().global_seed)
    return STAGES[args.stage]()


if __name__ == "__main__":
    sys.exit(main())

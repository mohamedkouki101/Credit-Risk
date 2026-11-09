# Decisions and plan deviations

Changes to the frozen plan (`phase1/`, `phase2/`) are recorded here with the reason. Newest last.

## 2026-11-08 · Step 0.3 · `src/credit_risk/models/` not created yet

- **Plan said:** step 0.3 creates the full PLAN §4 tree, including `src/credit_risk/models/`.
- **Conflict:** P-01 acceptance (and `scripts/check_prereg.py`) requires the `prereg-v1` tag commit to be **earlier than the first commit touching `src/credit_risk/models/`**. An empty `models/__init__.py` committed now would make that check fail permanently.
- **Decision:** create `src/credit_risk/models/` only after Kouki has tagged `prereg-v1` (step 1.5). All other folders from PLAN §4 exist.

## 2026-11-08 · Step 0.1 · No console-script entry point

- `uv init` adds `[project.scripts] credit-risk = "credit_risk:main"`. Removed: the plan's single entry point is `python -m credit_risk <stage>` (works the same on Windows), so a second one would only drift.

## 2026-11-09 · Step 0.4 · Primary data = full Kaggle file

- Kaggle page shows CC0, so per data plan §5.1 the full LC file is primary and Zenodo is the fallback/benchmark. Evidence, hash and caveats in `research/data_decision.md`.
- `kaggle` 2.2.5 is missing its `six` dependency; download ran with `uv run --with six`. Kouki chose to add `six` as a dev dependency (2026-11-09), so `uv run kaggle ...` works as is.

## 2026-11-08 · Step 0.2 · No `pandas<3` pin

- Roadmap E-01 allowed pinning `pandas<3` if OptBinning broke. It didn't: OptBinning 1.0.0 passes the smoke test on pandas 3.1.1 (Windows 11 and WSL2). See `research/env_spike.md`.

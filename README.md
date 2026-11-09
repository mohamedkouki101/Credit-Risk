# Credit default prediction: scorecard vs gradient boosting

A bank-style credit risk model on Lending Club data: a WOE logistic scorecard (champion) against monotone LightGBM (challenger), tested out of time, with imbalance handling, AUC/KS, SHAP and Reg B reason codes, a fairness audit, PSI monitoring and a live scoring app.

> **Status: 🟡 Setup** · Plan approved 2026-11-08 · Phase 0 in progress

## Progress

**Current step:** 0.5 Write the configs
**Next up:** 1.1 Download and load

| # | Phase | Status | Steps done | Exit criterion |
|---|---|---|---|---|
| 0 | Environment and gates | 🔵 In progress | 4 / 5 | Env locked, data licence decided, tests pass locally |
| 1 | Data | ⬜ Not started | 0 / 7 | Clean data, leakage tests, splits frozen, `prereg-v1` tagged |
| 2 | Baselines and champion scorecard | ⬜ Not started | 0 / 6 | Scorecard with PDO points, hand-checked example |
| 3 | Challenger, calibration, imbalance | ⬜ Not started | 0 / 5 | Calibrated LightGBM, imbalance method chosen by rule |
| 4 | Out-of-time evaluation | ⬜ Not started | 0 / 3 | OOT metrics with CIs, business view, stability |
| 5 | Explainability | ⬜ Not started | 0 / 5 | SHAP + reason codes, fidelity and stability tested |
| 6 | Fairness | ⬜ Not started | 0 / 4 | Fairness metrics, mitigation trade-off, legal note |
| 7 | Model risk | ⬜ Not started | 0 / 3 | Champion/challenger decision, PSI monitoring |
| 8 | Deployment | ⬜ Not started | 0 / 4 | API and app live with fallback |
| 9 | Docs and release | ⬜ Not started | 0 / 7 | Model card, validation report, `v1.0` |
| ✂ | Stretch (after v1.0) | ⏸ Parked | – | – |

Legend: ⬜ Not started · 🔵 In progress · ✅ Done · 🔴 Blocked · ⏸ Parked

Overall: **4 / 49 steps** · Full checklist in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)



## Key results

Filled in automatically from `results/*.csv` once the out-of-time run is done (step 4.1). No numbers are typed by hand.

| Model | OOT AUC (95% CI) | KS | Brier | Decision |
|---|---|---|---|---|
| M2 Scorecard (champion) | – | – | – | – |
| M4 Monotone LightGBM (challenger) | – | – | – | – |

## Build log

| Date | Steps | What changed |
|---|---|---|
| 2026-11-08 | Setup | Phase 1 research and Phase 2 plan added; implementation plan, CLAUDE.md and this README created |
| 2026-11-08 | 0.1 (E-01) | `uv init --package --vcs none`, Python pinned to 3.12, `src/credit_risk` package, `.gitignore`, import test passes |
| 2026-11-08 | 0.2 (E-01) | Deps locked in `uv.lock`; `scripts/smoke_env.py` passes on Windows 11 and WSL2; OptBinning works on pandas 3.1.1, so no `pandas<3` pin (R3 cleared); see `research/env_spike.md` |
| 2026-11-08 | 0.3 (E-02) | Skeleton per PLAN §4 (except `src/credit_risk/models/`, deferred until `prereg-v1`); `config.py`, `seed.py`, `log.py` (PLAN §7 schema), `python -m credit_risk` CLI with 10 stage stubs, Makefile, CI + slow workflows, pre-commit; 41 tests pass on Windows and WSL; ruff and mypy (strict) clean |
| 2026-11-09 | 0.4 (D-01) | Kaggle licence verified as CC0 (screenshots + API); PRIMARY = full LC file; `accepted_2007_to_2018Q4.csv.gz` downloaded, 392,582,231 bytes, SHA-256 `55c16f75…bd62a2d`, 151 columns, 2,260,701 lines; see `research/data_decision.md` |
| 2026-11-09 | 0.4 fix | Added `six` as a dev dependency (`kaggle` 2.2.5 imports it without declaring it); `uv run kaggle` works on Windows with no workaround |

## Repo map

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): ordered build checklist
- [phase2/PLAN.md](phase2/PLAN.md): approved plan (criteria, architecture, methods, tests, deployment)
- [phase2/docs/](phase2/docs/): roadmap, data plan, interview kit, learning path, sources
- [phase1/PHASE1_RESEARCH.md](phase1/PHASE1_RESEARCH.md): background research
- [research/decisions.md](research/decisions.md): plan deviations and why
- [research/env_spike.md](research/env_spike.md): locked versions and environment checks

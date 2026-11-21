# Credit default prediction: scorecard vs gradient boosting

[![CI](https://github.com/mohamedkouki101/Credit-Risk/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamedkouki101/Credit-Risk/actions/workflows/ci.yml)

A bank-style credit risk model on Lending Club data: a WOE logistic scorecard (champion) against monotone LightGBM (challenger), tested out of time, with imbalance handling, AUC/KS, SHAP and Reg B reason codes, a fairness audit, PSI monitoring and a live scoring app.

> **Status: 🟡 Building** · Plan approved 2026-11-08 · Phase 0 done, Phase 1 (Data) next

## Progress

**Current step:** 1.5 Pre-registration: 🔴 waiting for Kouki (commit + tag `prereg-v1`, see below)
**Next up:** 2.2 Naive and incumbent baselines (first code in `models/`: needs `prereg-v1`)

| # | Phase | Status | Steps done | Exit criterion |
|---|---|---|---|---|
| 0 | Environment and gates | ✅ Done | 5 / 5 | Env locked, data licence decided, tests pass locally |
| 1 | Data | 🔵 In progress | 6 / 7 | Clean data, leakage tests, splits frozen, `prereg-v1` tagged |
| 2 | Baselines and champion scorecard | 🔵 In progress | 1 / 6 | Scorecard with PDO points, hand-checked example |
| 3 | Challenger, calibration, imbalance | ⬜ Not started | 0 / 5 | Calibrated LightGBM, imbalance method chosen by rule |
| 4 | Out-of-time evaluation | ⬜ Not started | 0 / 3 | OOT metrics with CIs, business view, stability |
| 5 | Explainability | ⬜ Not started | 0 / 5 | SHAP + reason codes, fidelity and stability tested |
| 6 | Fairness | ⬜ Not started | 0 / 4 | Fairness metrics, mitigation trade-off, legal note |
| 7 | Model risk | ⬜ Not started | 0 / 3 | Champion/challenger decision, PSI monitoring |
| 8 | Deployment | ⬜ Not started | 0 / 4 | API and app live with fallback |
| 9 | Docs and release | ⬜ Not started | 0 / 7 | Model card, validation report, `v1.0` |
| ✂ | Stretch (after v1.0) | ⏸ Parked | – | – |

Legend: ⬜ Not started · 🔵 In progress · ✅ Done · 🔴 Blocked · ⏸ Parked

Overall: **12 / 49 steps** · Full checklist in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)



## Blocked / needs Kouki

**1.5 Pre-registration (P-01).** `PREREGISTRATION.md` is written and filled from `results/vintage_profile.csv`. Please:
1. Read `PREREGISTRATION.md` (criteria C1–C8 are PLAN §2 verbatim; the split counts are generated).
2. Commit it, together with the rest of the working tree.
3. `git tag prereg-v1` on that commit, then `git push` and `git push origin prereg-v1`.
4. Tell Claude the tag is pushed. `uv run python scripts/check_prereg.py` must then print `OK (tag prereg-v1 at …)`.

Until then: no code in `src/credit_risk/models/`, so 2.2 onward waits. Do not commit any file under `src/credit_risk/models/` in the tagged commit, or `check_prereg.py` fails for good.

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
| 2026-11-09 | 0.5 | All 8 configs written (`base` with data source + SHA-256, `target`, `splits`, `features`, `monotone`, `scorecard`, `reason_text`, `business`); `load_all_configs()` validates each file and the cross-file rules (disjoint feature groups, every loan status in one class, contiguous OOT splits, backtests never reach 2015, monotone and reason families only on whitelisted features); 22 features whitelisted; 67 tests pass on Windows |
| 2026-11-09 | 1.1 (D-02) | `data/download.py`: `download("kaggle"|"zenodo")` with SHA-256/md5 + size check, `.part` file renamed only after it verifies, `DataIntegrityError` on mismatch; `data/load.py::load_raw()` reads 37 allowed columns with explicit dtypes (post-origination columns never loaded); Zenodo file name added to `base.yaml`; 84 fast tests pass |
| 2026-11-09 | 1.2 (D-03) | `data/target.py`: `build_target` adds `y` (set only inside the population), `segment`, `exclusion_reason`, `policy_flag`; T-06 (unknown or null `loan_status` raises); `vintage_profile` + `render_profile_md` per issue quarter with the >1% indeterminate flag. The `data` stage that writes the profile is wired in 1.3 (needs `clean.py`; see `research/decisions.md`) |
| 2026-11-09 | 1.3 (D-04) | `clean.py` (rules 5.4, drop log), `features.py` (`derive`, `model_frame`, T-01 whitelist + narrowed leakage regex, T-04 month arithmetic), pandera `LoanSchema`, `sql/vintage_bad_rate.sql` cross-checked against pandas; `python -m credit_risk data` runs end to end: 2,260,701 raw rows → 2,260,668 clean (33 footer rows), 620,783 labelled loans, 147 indeterminate, no vintage flagged; outputs in `results/*.csv` and `reports/data_profile.md` |
| 2026-11-09 | 1.4 (D-05) | `data/audit.py`: per feature (train window only) missing rate by target, L6 flag, IV on train and per issue year, mechanical keep/drop; `research/feature_audit.md` generated by the `data` stage: 20 keep, 2 drop (`term`, `application_type`, constant in core), no L6 or IV > 0.5 flags |
| 2026-11-09 | 1.5 (P-01), Claude part | `PREREGISTRATION.md` (PLAN §2 copied verbatim + generated split counts and config hashes via `scripts/fill_prereg.py`; train 175,426 labelled loans / 22,157 bad, valid 162,570 / 22,315, OOT 282,787 / 42,089); `scripts/check_prereg.py` (read-only git) added to CI with full-history checkout. Waiting for Kouki's `prereg-v1` tag |
| 2026-11-09 | 2.1 (EV-01) | `evaluation/metrics.py`: `auc`, `gini`, `ks`, `pr_auc`, `brier`, `log_loss`, `calibration_slope_intercept` (Newton fit on the log-odds scale), `calibration_table`, paired `delong_test`, `paired_bootstrap`, `bootstrap_ci`; T-10 (KS = `ks_2samp` = max(TPR−FPR), also with ties), T-11 (monotone invariance via hypothesis, Gini = 2·AUC−1, constant score = 0.5), closed-form two-Gaussian AUC/KS, DeLong SE vs bootstrap; done early because 1.7 needs it |
| 2026-11-09 | 1.7 (T-05) | `tests/synthetic/`: 200k-row known-logit data; parts 1 (LR recovers coefficients ±0.05), 3 (noise IV < 0.02), 4 (calibration slope in [0.95, 1.05]) and 6 (closed-form AUC 0.760 / KS 0.383) pass; parts 2, 3-selection, 5, 7 are strict xfails until the model code exists; 3 s, not slow |
| 2026-11-10 | 1.6 (D-06) | `data/split.py`: `make_splits` (out-of-time on issue month), T-02 (strict date order, no shared id), `python -m credit_risk features` writes `data/processed/{train,valid,test,monitor}.parquet` (monitor without `y`) and `results/split_summary.csv`, fills the split table in `reports/data_profile.md`: train 175,426 (12.63% bad), valid 162,570 (13.73%), OOT 282,787 (14.88%), monitor 1,253,029 unlabelled. Fixed the 2007Q2 omission in the registered train count |

## Repo map

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): ordered build checklist
- [phase2/PLAN.md](phase2/PLAN.md): approved plan (criteria, architecture, methods, tests, deployment)
- [phase2/docs/](phase2/docs/): roadmap, data plan, interview kit, learning path, sources
- [phase1/PHASE1_RESEARCH.md](phase1/PHASE1_RESEARCH.md): background research
- [research/decisions.md](research/decisions.md): plan deviations and why
- [research/env_spike.md](research/env_spike.md): locked versions and environment checks
- [research/data_decision.md](research/data_decision.md): data source, licence evidence, file hash
- [configs/](configs/): all settings (target, splits, feature whitelist, scorecard, business assumptions)

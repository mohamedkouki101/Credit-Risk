# Implementation plan

The step-by-step build plan for Claude Code. It turns the approved Phase 2 plan into an ordered checklist you can work through one step at a time.

- **Source of truth for *what* and *why*:** [`phase2/PLAN.md`](phase2/PLAN.md) and the task cards in [`phase2/docs/03_roadmap.md`](phase2/docs/03_roadmap.md). Each step below gives its roadmap ID (e.g. `M-05`). Read that card before you start the step: it has the exact acceptance test.
- **Data rules:** [`phase2/docs/02_data_plan.md`](phase2/docs/02_data_plan.md) (whitelist, target, splits, leakage L1–L9, bias B1–B7).
- **Background research:** [`phase1/PHASE1_RESEARCH.md`](phase1/PHASE1_RESEARCH.md) and [`phase2/RESEARCH_INVENTORY.md`](phase2/RESEARCH_INVENTORY.md).
- **Status:** plan approved 2026-11-08. Nothing built yet.

## How to work through this plan

1. Take the **first unchecked step** in order. Do not skip ahead; ★ steps are on the critical path.
2. Read its roadmap card, then implement it with tests.
3. Run the step's **Done when** check. Run `uv run pytest -m "not slow"` and `uv run ruff check .` before ticking anything.
4. Tick the box here (`- [x]`), update the **README progress table** and add one line to the README **Build log** (date, step IDs, what changed, any key number).
5. **No git.** Kouki handles the repository and all git work. Do not run `git init`, `add`, `commit`, `tag`, `push`, `remote`, `branch`, `checkout`, `reset` or `stash`, and do not create GitHub repos. Read-only commands (`git status`, `git log`, `git diff`) are fine. When a step finishes, end with a one-line suggested commit message (`<STEP-ID>: <summary>`) for Kouki to use.
6. Steps marked **👤 USER** need Kouki (an account, a login, a key, a git action or a click in a web UI). Stop at those, explain exactly what to do, and work on the next step that doesn't depend on it.
7. Stop and ask before: touching the OOT test set outside `make final`, changing anything frozen in `PREREGISTRATION.md`, adding a dependency not listed in E-01, or starting any stretch task.

**Local only:** Claude Code works in this folder on Kouki's machine. Kouki owns the GitHub repo, all git operations, and pushing to trigger CI.

**Platform note:** Kouki works on Windows. Every pipeline stage must run as `uv run python -m credit_risk <stage>` (a small CLI in `src/credit_risk/__main__.py`), and the `Makefile` targets are thin wrappers around those commands. So the project works with or without `make`. Use `pathlib` for paths, Python `hashlib` instead of `sha256sum`, and no shell-only tricks in scripts.

---

## Phase 0: Environment and gates

**Exit:** reproducible env, licence decision recorded, CI green, `prereg-v1` tagged.

- [x] **0.1 Initialise the Python package** (part of E-01)
  - `uv init --package --name credit-risk --vcs none` in the existing folder (keep `phase1/`, `phase2/`, the plan files; `--vcs none` stops uv from running `git init`). Package import name `credit_risk` under `src/`. `uv python pin 3.12`.
  - Write a `.gitignore` file (a plain file, for Kouki's git use) covering Python, `.venv/`, `data/raw/`, `data/interim/`, `data/processed/`, `.env`, `*.parquet` outside `models/`, `kaggle.json`.
  - **Done when:** `uv run python -c "import credit_risk"` works.

- [x] **0.2 ★ Dependency lock and environment smoke test** (E-01)
  - Add runtime deps: `pandas pyarrow numpy scipy scikit-learn lightgbm optbinning shap fairlearn pandera pydantic pyyaml duckdb matplotlib fastapi uvicorn streamlit httpx imbalanced-learn joblib jinja2`.
  - Add dev deps: `pytest pytest-cov hypothesis ruff mypy pre-commit`.
  - Write `scripts/smoke_env.py`: OptBinning `BinningProcess` + `Scorecard` on `make_classification` data in a pandas DataFrame; LightGBM with `monotone_constraints`; SHAP `TreeExplainer`. Time each step.
  - If OptBinning breaks on pandas 3, pin `pandas<3` and record why (risk R3).
  - Write `research/env_spike.md`: versions, timings, RAM, any incompatibility.
  - **Done when:** `uv run python scripts/smoke_env.py` exits 0; `uv.lock` exists.

- [x] **0.3 Repo skeleton, config system, CI, pre-commit** (E-02)
  - Create the tree from PLAN §4 (empty modules with docstrings are fine): `configs/`, `src/credit_risk/{data,models,evaluation,explain,fairness,monitoring,reporting}/`, `api/`, `app/`, `models/`, `results/`, `reports/`, `research/`, `notebooks/`, `scripts/`, `sql/`, `tests/{unit,synthetic,regression,api}/`, `data/README.md`.
  - `src/credit_risk/config.py`: pydantic models that load `configs/*.yaml`. `src/credit_risk/seed.py::set_all_seeds()`. `src/credit_risk/log.py`: experiment-log writer with the schema in PLAN §7.
  - `src/credit_risk/__main__.py` CLI with stages `data features train evaluate explain fairness monitor report final export` (stubs for now). `Makefile` wraps them plus `setup test lint api app`.
  - `.github/workflows/ci.yml`: `uv sync` → `ruff check` → `mypy src` → `pytest -m "not slow"`. `.github/workflows/slow.yml` (manual trigger, filled in later).
  - `.pre-commit-config.yaml`: ruff, end-of-file fixer, block files > 5 MB. Write the config only; Kouki runs `pre-commit install` in the repo.
  - `pyproject.toml`: register the `slow` pytest marker; ruff and mypy settings.
  - **Done when:** one trivial test passes locally; ruff and mypy run clean; the CI and pre-commit YAML files parse. Tell Kouki the CI workflow runs once Kouki pushes, and where to put the CI badge in the README.

- [x] **0.4 ★ 👤 USER: Kaggle licence and access gate** (D-01)
  - Kouki: log in to Kaggle, open `kaggle.com/datasets/wordsforthewise/lending-club`, screenshot the licence + file list into `research/licence_evidence/`, create an API token (`kaggle.json` in `%USERPROFILE%\.kaggle\`, never in the repo).
  - Claude Code: download with `uv run kaggle datasets download wordsforthewise/lending-club -p data/raw` (add `kaggle` as a dev dep), compute SHA-256 with Python, write `research/data_decision.md`: PRIMARY = full LC file or Zenodo, with evidence and hash.
  - Rule: licence not clearly permissive → PRIMARY = Zenodo (data plan §5.3b).
  - **Done when:** `research/data_decision.md` exists with screenshot path and hash.

- [ ] **0.5 Write the configs** (inputs for D-02…D-06 and P-01)
  - `configs/base.yaml` (paths, `global_seed: 42`, `data_source`), `target.yaml` (status mapping from data plan §5.3a), `splits.yaml` (§5.5 boundaries), `features.yaml` (§5.2 whitelist + excluded columns each with a reason), `monotone.yaml` (roadmap M-06 directions), `scorecard.yaml` (PDO 20, 600 at 50:1, binning params from M-03), `reason_text.yaml` (placeholder families), `business.yaml` (LGD source, margin grid {0.05, 0.10, 0.15}, approval rates {0.6, 0.7, 0.8}).
  - **Done when:** `config.py` loads all of them and a unit test asserts they validate.

---

## Phase 1: Data

**Exit:** validated parquet, whitelist enforced by tests, data profile, splits frozen, `prereg-v1` tagged.

- [ ] **1.1 ★ Download and load** (D-02)
  - `data/download.py::download(source)`; Zenodo streams and checks md5 `b019384d6bc65bf2a3e839362e4ff502`; Kaggle checks the hash recorded in 0.4. `DataIntegrityError` on mismatch.
  - `load_raw()` reads only whitelist + outcome + fairness columns (`usecols`, explicit dtypes).
  - **Done when:** unit test with a corrupted temp file raises `DataIntegrityError`; `data/README.md` explains how to get the data.

- [ ] **1.2 ★ Target construction and vintage profile** (D-03)
  - `data/target.py::build_target(df, cfg)` → adds `y`, `segment`, `exclusion_reason` exactly per data plan §5.3a (individual, 36-month, issued 2007-06…2015-12; bad/good/indeterminate mapping).
  - `reports/data_profile.md` per issue quarter: rows, bad rate, indeterminate count, missingness. Flag any vintage with > 1% indeterminate.
  - **Done when:** T-06 passes (unknown `loan_status` raises); `python -m credit_risk data` regenerates the profile.

- [ ] **1.3 ★ Cleaning, derived features, schema, SQL** (D-04)
  - `data/clean.py` (rules §5.4), `data/features.py::derive()` (`fico_mid`, `credit_age_months` from `issue_d`, log1p transforms), pandera `LoanSchema` in `data/schema.py`.
  - `sql/vintage_bad_rate.sql` run through DuckDB for the vintage tables.
  - **Done when:** T-01 (whitelist + leakage regex), T-04 (date arithmetic incl. leap years) and pandera validation pass; clean row count = raw − logged drops (asserted).

- [ ] **1.4 Feature leakage and missingness audit** (D-05, parallel)
  - Per feature: missing rate by target, train IV, IV per vintage; flag rule L6.
  - **Done when:** `research/feature_audit.md` has a keep/drop decision for every whitelist feature.

- [ ] **1.5 ★ 👤 USER + Claude: Pre-registration** (P-01) — **must happen before any file in `src/credit_risk/models/` gets real code**
  - Claude Code: copy PLAN §2 into `PREREGISTRATION.md`, fill split boundaries and vintage counts from 1.2. Write `scripts/check_prereg.py` (read-only git: asserts the `prereg-v1` tag's commit predates the first commit touching `src/credit_risk/models/`) and add it to CI.
  - Then **stop** and ask Kouki to commit `PREREGISTRATION.md`, run `git tag prereg-v1` and push the tag. Do not write model code until Kouki confirms.
  - **Done when:** Kouki confirms the tag exists; `uv run python scripts/check_prereg.py` passes.

- [ ] **1.6 ★ Split materialisation** (D-06)
  - `data/split.py::make_splits(df, cfg)`; write `data/processed/{train,valid,test,monitor}.parquet`. Bad rate per split into the data profile.
  - **Done when:** T-02 passes (strict date order, no ID overlap).

- [ ] **1.7 Synthetic known-answer test** (T-05) — write it now so every later model is checked against it
  - `tests/synthetic/` per PLAN §9 (200k rows, known logit, Bayes AUC, closed-form AUC 0.760 / KS 0.383 check). Mark `slow` if it takes > 30 s.
  - **Done when:** the metric parts pass now; the model parts are marked `xfail` until Phase 2–3 code exists, then turned on.

---

## Phase 2: Baselines and the champion scorecard

**Exit:** scorecard with PDO points, hand-checked example, validation metrics logged.

- [ ] **2.1 ★ Metric library** (EV-01; moved early because every later step needs it)
  - `evaluation/metrics.py`: `auc, gini, ks, pr_auc, brier, log_loss, calibration_table, calibration_slope_intercept, delong_test, paired_bootstrap`.
  - **Done when:** T-10 (KS = `ks_2samp` = max(TPR−FPR)), T-11 (AUC invariances, Gini = 2·AUC−1) pass.

- [ ] **2.2 ★ Naive and incumbent baselines** (M-01)
  - M0 = constant train bad rate; M1 = `sub_grade` ordinal as a score. Log both with bootstrap CIs to `results/experiments.jsonl`.
  - **Done when:** M0 AUC == 0.5 test passes; both runs are in the log.

- [ ] **2.3 Literature reproduction on Zenodo** (M-02, parallel, not a gate)
  - Target AUC 0.6575 ± 0.01 (Sanz-Guerrero & Arroyo). Write `research/reproduction.md` either way.

- [ ] **2.4 ★ Monotonic WOE binning** (M-03)
  - `models/scorecard.py::fit_binning()` with `monotonic_trend="auto_asc_desc"`, `min_bin_size=0.05`, missing as a special bin, IV selection [0.02, 1]. Export `reports/binning/*.csv`. Flag IV > 0.5 for a leakage check.
  - **Done when:** T-03 (fit on train only) and T-07 (monotone WOE) pass.

- [ ] **2.5 ★ Logistic regression on WOE** (M-04)
  - L2 LR, `C` in [0.01, 0.1, 1, 10] picked on validation log-loss. Check coefficient signs (document OptBinning's WOE sign convention in the docstring), VIF > 5 flagged.
  - **Done when:** all coefficients share the expected sign (violators dropped and logged).

- [ ] **2.6 ★ PDO scaling and score bands** (M-05)
  - `scale_points()` with factor = PDO/ln 2, offset = base − factor·ln(base_odds); per-attribute points; 10 score bands with observed bad rate and mean PD.
  - Write `docs/scorecard_example.md`: one applicant computed by hand.
  - **Done when:** T-08 passes (+20 points doubles odds; points sum to total; hand example matches to the point).

---

## Phase 3: Challenger, calibration and imbalance

**Exit:** calibrated challenger; imbalance arm chosen by the pre-registered rule.

- [ ] **3.1 ★ LightGBM challengers** (M-06)
  - M3 unconstrained; M4 monotone per `configs/monotone.yaml`; methods `intermediate` and `advanced`; bounded search (≤ 30 points), early stopping on validation log-loss, `seed=42, deterministic=True, force_row_wise=True`.
  - **Done when:** T-09 (no monotonicity violations on a grid) passes; every run logged.

- [ ] **3.2 Rolling-origin backtest** (M-07)
  - 3 folds (data plan §5.5) for M2 and M4, binning refit per fold. Table of AUC/KS/Brier per fold, mean ± sd.

- [ ] **3.3 ★ Calibration** (M-08)
  - `CalibratedClassifierCV(FrozenEstimator(model))`, sigmoid vs isotonic, chosen by 5-fold cross-fitting inside the validation year. Report curve, slope, intercept, Brier.
  - **Done when:** method recorded; test asserts sigmoid keeps rank order.

- [ ] **3.4 ★ Imbalance comparison** (M-09)
  - Arms A natural + cutoff, B class weights + recal, C undersample + recal, D SMOTENC + recal, for M2 and M4.
  - `select_arm()` implements the pre-registered rule (lowest validation log-loss among arms not significantly worse on AUC; tie → A) with a unit test on a toy table.
  - **Done when:** `reports/imbalance.md` written by code.

- [ ] **3.5 Ablations and sensitivity runs on validation** (PLAN §7: A-01…A-07, SA-01, SA-03, SA-04)
  - Each one is a logged run. OOT only for the final pair.

---

## Phase 4: Out-of-time evaluation

**Exit:** OOT results with CIs, business view, stability.

- [ ] **4.1 ★ The once-only OOT run** (EV-02) — **ask Kouki before running**
  - `python -m credit_risk final`: writes `results/final.lock` (config hash), scores test for M0–M4, writes `results/oot_metrics.csv`, DeLong M4 vs M2, 2,000 paired bootstraps for ΔAUC/ΔKS/ΔBrier.
  - Check criterion C2: any OOT AUC > 0.80 → stop, write a leakage audit, report nothing else.
  - **Done when:** lock written; rerun with a changed config fails.

- [ ] **4.2 Business view** (EV-03)
  - Realised LGD from train charge-offs; EL = PD × LGD × EAD; approval vs bad-rate curves; table at 60/70/80% approval; margin sensitivity (SA-05).

- [ ] **4.3 Stability** (EV-04)
  - Per-quarter OOT metrics; adversarial validation; score + feature PSI dev→OOT and dev→monitoring with simulated critical values.
  - **Done when:** T-12 (PSI) passes; `reports/stability.md` generated.

---

## Phase 5: Explainability

- [ ] **5.1 Global explanations** (X-01): scorecard IV/points; TreeSHAP interventional, log-odds, 500-row background; beeswarm + bar; Spearman IV vs SHAP rank. **Done when:** additivity test (1e-6) passes.
- [ ] **5.2 ★ Reason codes, two methods** (X-02): scorecard points-shortfall (Reg B methods 1 and 2) and SHAP with reason families; text from `configs/reason_text.yaml`. **Done when:** ≤ 4 reasons, only model features, deterministic (tests).
- [ ] **5.3 ★ Fidelity test** (X-03): 2,000 declined OOT applicants, top reason vs random feature. Threshold ≥ 80% (pre-registered C7).
- [ ] **5.4 Explanation stability** (X-04): 10 seeds, Kendall's W, Jaccard of top-4 reason sets.
- [ ] **5.5 Scorecard vs SHAP comparison** (X-05). **Done when:** T-13 (LR-SHAP = centred contributions) passes.

---

## Phase 6: Fairness

- [ ] **6.1 👤 USER + Claude: proxy groups** (F-01): Kouki gets a free Census API key (stored in `.env`, never committed). ACS B03002 by ZCTA → ZIP3 minority share → join to `zip_code[:3]`. Coverage ≥ 95% or write the fallback in `research/fairness_proxy.md`.
- [ ] **6.2 ★ Fairness metrics** (F-02): Fairlearn `MetricFrame` for M2 and M4 at the cutoff, plus `demographic_parity_ratio`, `equalized_odds_difference`, calibration within groups, bootstrap CIs, at 3 cutoffs (SA-06). South German Credit by age (< 25, ≥ 25, ≥ 62 where n allows).
- [ ] **6.3 Mitigation trade-off** (F-03): ThresholdOptimizer (demo only, disparate-treatment caveat), ExponentiatedGradient on WOE LR, small LDA search. Figure + AUC-cost table.
- [ ] **6.4 Legal context page** (F-04): `docs/fairness_legal_context.md`, every claim linked and dated, "not legal advice". Re-check the Reg B litigation status before writing.

---

## Phase 7: Model risk

- [ ] **7.1 ★ Champion vs challenger memo** (MR-01): a script applies criterion C3 mechanically and prints the decision; `reports/champion_challenger.md`.
- [ ] **7.2 Monitoring job** (MR-02): `monitoring/psi.py::monitor(batch)` on 2016–2018; injected-shift test.
- [ ] **7.3 Reject-inference section** (MR-03): half page, Kozodoi et al. kickout measure.

---

## Phase 8: Deployment

- [ ] **8.1 ★ Artefact export** (DEP-01): `models/*.joblib`, `bins.json`, `background.parquet`, `manifest.json` with hashes; < 50 MB. **Done when:** T-14 (bit-for-bit rescoring of 100 rows) passes.
- [ ] **8.2 ★ FastAPI service** (DEP-02): `POST /score`, `GET /health`, `GET /model-card`; Dockerfile on `python:3.12-slim`. **Done when:** TestClient tests pass (200, 422, schema).
- [ ] **8.3 ★ Streamlit app** (DEP-03): form + example applicants + result card + tabs; API with 3 s timeout, in-process fallback. **Done when:** Playwright smoke test passes with the API down.
- [ ] **8.4 👤 USER + Claude: deploy** (DEP-04): Kouki creates the Render and Streamlit Community Cloud services (steps in PLAN §10); Claude Code writes `scripts/smoke_live.py` and the README badges. **Done when:** both URLs respond and the live smoke test passes.

---

## Phase 9: Documentation and release

- [ ] **9.1 ★ Model card** (DOC-01): `MODEL_CARD.md`, Mitchell et al. + SR 26-2 items, generated from a template.
- [ ] **9.2 Results report** (DOC-02): all tables and figures from PLAN §11.
- [ ] **9.3 ★ Validation report** (DOC-03): `VALIDATION_REPORT.md` on the three SR 26-2 pillars.
- [ ] **9.4 ★ Portfolio README** (DOC-04): expand the README to the PLAN §11 outline, numbers filled by `python -m credit_risk report`. Keep the progress table, moved to the bottom under "Build log". **Done when:** T-15 (README numbers match `results/*.csv`) passes.
- [ ] **9.5 Interview kit with real numbers** (DOC-06): fill the blanks in `phase2/docs/04_interview_kit.md` from results only.
- [ ] **9.6 Regression goldens**: `tests/regression/golden_metrics.json`; slow CI job on a 5% sample. Any later change to the golden file is flagged to Kouki with the reason, for the commit message.
- [ ] **9.7 ★ 👤 USER + Claude: Release** (FINAL): Claude Code records the demo GIF and checks the v1.0 checklist; Kouki tags `v1.0`.

---

## ✂ Cut line

Stretch tasks S-01…S-09 (see the table at the end of `phase2/docs/03_roadmap.md`) start only after `v1.0` is tagged and Kouki says so.

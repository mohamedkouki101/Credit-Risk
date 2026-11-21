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

## 2026-11-09 · Step 0.5 · Config details settled from the real file

- **`loan_status` values:** the v3 file has exactly 9 non-null statuses. All are listed explicitly in `target.yaml`: Current, In Grace Period, Late (16-30 days) and Late (31-120 days) are indeterminate, as data plan 5.3a says ("any other status"). The 33 null-status rows are footer artefacts removed by cleaning rule 5.4.4.
- **`emp_length`:** the plan says `"n/a"` -> missing, but the file has true nulls (146,940) and no `"n/a"`. Same outcome (missing); the transform note says so.
- **`term` and `application_type`** stay in the whitelist as in data plan 5.2 although they are constant in the core population. Binning drops constant features, so this costs nothing and keeps S-04 open.
- **Monotone directions** are exactly roadmap M-06. **Reason families** are placeholders (`placeholder: true`) until X-02.
- **Hashes are quoted in YAML:** an unquoted all-digit hash parses as an integer.

## 2026-11-08 · Step 0.2 · No `pandas<3` pin

- Roadmap E-01 allowed pinning `pandas<3` if OptBinning broke. It didn't: OptBinning 1.0.0 passes the smoke test on pandas 3.1.1 (Windows 11 and WSL2). See `research/env_spike.md`.

## 2026-11-09 · Steps 1.1-1.2 · Loader and target details

- **Zenodo file name in `base.yaml`:** `LC_loans_granting_model_dataset.csv` (from the Zenodo API, md5 matches the plan). Needed to build the download URL; the plan only named the md5.
- **`load_raw` also reads the benchmark columns** (`grade`, `sub_grade`, `int_rate`, `installment`): the incumbent baseline M-01 needs them. They stay out of `X` (T-01). 37 columns in total; post-origination columns are never loaded.
- **Pipeline order is load -> clean -> target -> derive.** `build_target` needs parsed `issue_d`, stripped `term` and no footer rows, which are cleaning rules 5.4.1-5.4.4 (D-04). So the `data` CLI stage that regenerates `reports/data_profile.md` is wired in step 1.3, once `clean.py` exists. Step 1.2 delivers `build_target`, T-06 and the profile functions, tested on hand-made frames.
- **`y` is set only inside the modelling population** (Individual, 36-month, 2007-06..2015-12, determinate status). Rows outside keep `y = <NA>` plus an `exclusion_reason`, so the 2016-2018 monitoring rows can never carry a label. A `policy_flag` column is added for SA-04.

## 2026-11-09 · Step 1.3 · Cleaning and T-01 details

- **T-01 leakage regex narrowed.** The plan's regex `pymnt|rec_|recover|settle|hardship|out_prncp|last_` also matches four whitelisted application-time bureau counts: `inq_last_6mths`, `mths_since_last_delinq`, `mths_since_last_record` (via `last_`) and `pub_rec_bankruptcies` (via `rec_`). Used `pymnt|(?<!pub_)rec_|recover|settle|hardship|out_prncp|^last_`. A test pins that, on all 151 raw columns, the only columns the plan's regex catches and ours does not are those 4 plus 4 other application-time counts that are not whitelisted (`inq_last_12m`, `mths_since_last_major_derog` and two `sec_app_*`). Every post-origination column is still caught.
- **Missing `earliest_cr_line` is kept** (credit age missing), only a present-but-unparsable value drops the row. Rule 5.4.1 is about parse failures. The real file has none of either kind that matter: 0 rows dropped by rule 5.4.1.
- **`annual_inc` < 0, `loan_amnt` <= 0 and `fico_mid` outside [300, 850] are not repaired**: the pandera schema rejects them, so the stage fails loudly. The real file passes.
- **Pipeline order is load -> clean -> target -> derive -> validate**, all in the `data` stage. It also writes `results/{vintage_profile,cleaning_log,population_counts}.csv` so the docs can quote numbers from files.
- **Memory:** the full `data` stage peaks at about 5.4 GB RAM and takes about 45 s (WSL2).

## 2026-11-09 · Step 1.4 · Feature audit rules

- **L6 gets a 1% materiality floor:** flagged only if the larger missing rate is at least 1% and more than 3x the smaller (or the smaller is 0). Without it, a few missing cells in one class would flag a feature. The 3x ratio is the plan's (Unverified) heuristic.
- **Decisions are mechanical** (first rule that fires): constant on train -> drop; L6 -> drop; IV > 0.5 (`iv_leakage_flag`) -> drop until audited; else keep. The audit reads train-window rows only.
- **Audit IV uses a coarse binning** (up to 10 quantile bins, missing as its own bin). It is a screening number, not the scorecard's IV (M-03 uses OptBinning).
- **Result:** 20 keep, 2 drop (`term`, `application_type`: constant in the core population, as expected from 0.5). No L6 flag and no IV above 0.5. Per-year IVs for 2007 are inflated by the small sample (a few hundred loans).

## 2026-11-09 · Steps 1.5, 2.1, 1.7 · Order and pre-registration details

- **2.1 done before 1.7:** T-05's metric parts need `evaluation/metrics.py` (EV-01, "can start early"). Neither touches `src/credit_risk/models/`.
- **`PREREGISTRATION.md` registers counts, not rates:** the generated table has labelled loans, bads and indeterminates per split, plus SHA-256 of every config (LF-normalised so Windows and Linux agree). No outcome rate or metric of the OOT year is registered.
- **`check_prereg.py` also guards the frozen configs:** if `splits.yaml`, `target.yaml` or `features.yaml` differ from the tag, it fails unless `PREREGISTRATION.md` has an entry under "## Amendments" naming the file. Its decision logic is a pure function tested with fakes, so no test runs a state-changing git command. CI checkout now uses `fetch-depth: 0` (tags and history).
- **T-05 model parts are strict xfails** that import the future `credit_risk.models.scorecard` / `credit_risk.models.lgbm`. When that code lands they XPASS (= fail) and must be switched on, with the call adjusted to the real API.
- **`LogisticRegression(C=np.inf)`** instead of `penalty=None` (deprecated in scikit-learn 1.8).

## 2026-11-10 · Step 1.6 · Splits, and a fix to the registered counts

- **Done before the `prereg-v1` tag.** The roadmap lists P-01 as a dependency of D-06 because the splits are frozen at the tag. `split.py` only reads `configs/splits.yaml` (hash-recorded in `PREREGISTRATION.md`) and writes no model code, so nothing is decided before registration. Step 2.2 (first code in `src/credit_risk/models/`) still waits for the tag.
- **Monitoring set** = issued 2016-01 to 2018-12, Individual applications, any term (data plan 5.5 says "all terms"). Its parquet has no `y`: the outcomes are not mature.
- **Bug fixed before the tag:** `split_counts` counted a quarter only if it *started* inside a window, so 2007Q2 (June 2007 loans only: 24 loans, 3 bad) was missing from the train count in `PREREGISTRATION.md` (175,402 instead of 175,426). Quarters now count if they overlap the window. A test asserts the registered counts equal `results/split_summary.csv`.

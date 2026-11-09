# Data plan (PLAN §5 in full)

## 5.1 Sources and licences

| Role | Dataset | Access | Licence | Status | Source · confidence |
|---|---|---|---|---|---|
| **Primary (gated)** | Lending Club accepted loans 2007–2018Q4, ~2.26M × 151 (Kaggle `wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz`) | Free Kaggle account + API token, `kaggle datasets download` | Tagged CC0 according to a secondary catalogue. **Not verified on the Kaggle page itself** | Gate D-01 | [Catalogue](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html) · **Medium**; [data dictionary (figshare)](https://figshare.com/articles/dataset/Lending_club_dataset_description/20016077) · **Medium** |
| **Fallback + benchmark** | Zenodo "Lending Club loan dataset for granting models" v0.1, 1,347,681 rows × 14 columns | Direct HTTPS download, md5 `b019384d6bc65bf2a3e839362e4ff502` | CC BY 4.0 | Verified | [Zenodo](https://zenodo.org/records/11295916) · **High** |
| Rejects (stretch) | LC rejected loans, ~27M × 9 (same Kaggle dataset) | Kaggle | As above | Gate D-01 | Catalogue · **Medium** |
| Fairness side-study | South German Credit (corrected German Credit), 1,000 × 21 | UCI download | CC BY 4.0 | Verified | [UCI 573](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) · **High** |
| Fairness proxy (core, gated) | US Census ACS 5-year race/ethnicity by ZCTA, aggregated to 3-digit ZIP | Census API (free key) | US government work, public domain (**Unverified**) | Spike F-01 | **Unverified** |

**Gate D-01 decision rule.** On your machine, open the Kaggle dataset page and screenshot the licence field into `research/licence_evidence/`.
- If it shows CC0 or another licence allowing research use, and the download works, the full file is primary.
- If not, Zenodo becomes primary. The target-window rules then switch to the Zenodo variant (§5.3b), and the README states the narrower feature set.

In both cases **raw data is never committed**. `make data` downloads it and checks a hash.

## 5.2 Schema: the application-time feature whitelist (full file)

A feature is allowed only if it was known **at the application date**. Everything not listed is excluded by default. The whitelist lives in `configs/features.yaml`, and a test enforces it (T-01).

| Group | Columns (raw name → model name) | Transform |
|---|---|---|
| Loan request | `loan_amnt`, `purpose`, `term` (constant in core, see 5.3) | `purpose`: merge categories under 1% share into `other` |
| Applicant | `emp_length` (ordinal 0–10, "n/a" → missing), `home_ownership` (OTHER/NONE/ANY → OTHER), `annual_inc` (log1p), `verification_status`, `application_type` (core filters to Individual) | |
| Affordability | `dti` (values below 0 or above 100 → missing, flag) | |
| Bureau | `fico_range_low`, `fico_range_high` → `fico_mid` = mean; `earliest_cr_line` → `credit_age_months` = months(issue_d − earliest_cr_line); `inq_last_6mths`, `delinq_2yrs`, `mths_since_last_delinq` (missing → "never" bin), `mths_since_last_record`, `open_acc`, `total_acc`, `pub_rec`, `pub_rec_bankruptcies`, `revol_bal` (log1p), `revol_util` (strip "%", over 150 → missing), `mort_acc` | Missing kept as its own WOE bin (scorecard) or native NaN (LightGBM) |
| **Fairness only, never a model input** | `addr_state`, `zip_code` (3-digit ZIP, e.g. "940xx") | Used only to build proxy groups (F-01) |
| **Incumbent benchmark only** | `grade`, `sub_grade`, `int_rate`, `installment` | `sub_grade` → ordinal 1–35 as "LC's own score" (M-01). Never a feature: these are outputs of LendingClub's own risk model, and `installment` is a function of `int_rate` |
| **Outcome and loss computation only** | `loan_status`, `last_pymnt_d`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `funded_amnt` | Used to build the target and the realised LGD. Leakage test T-01 asserts none of these reach `X` |

Explicitly excluded, with the reason recorded in `configs/features.yaml`:
- **Post-origination fields:** `last_fico_range_*`, `total_pymnt*`, `out_prncp*`, `last_credit_pull_d`, `hardship_*`, `debt_settlement_*`, `settlement_*`, `next_pymnt_d`, `pymnt_plan`, `collections_12_mths_ex_med` (timing uncertain, so excluded conservatively).
- **Fields added from late 2015** (e.g. `open_acc_6m`, `all_util`, `il_util`): mostly missing in training vintages, and their presence leaks the date.
- **Free text** (`emp_title`, `title`, `desc`): text features are out of scope (Phase 1, low payoff).
- `initial_list_status`, `policy_code`, `funded_amnt_inv`: investor-side fields.

Zenodo variant: `revenue`, `dti_n`, `loan_amnt`, `fico_n`, `experience_c`, `emp_length`, `purpose`, `home_ownership_n` (+ `addr_state`/`zip_code` for fairness only).

## 5.3 Target, default definition and observation window

**Why not the Basel definition?** Basel IRB defines default as 90+ days past due or unlikely to pay ([BCBS CRE36](https://www.bis.org/committees/bcbs/basel-framework/81373/chapter.pdf) · **High**). The Lending Club file has no monthly delinquency history, only the loan status at the snapshot. So the target is a **charge-off** definition, which is later and stricter than 90 DPD. Predicted PDs are therefore "charge-off over the loan's life" PDs and will sit below a Basel 90-DPD PD. This goes in the model card limitations.

**(a) Primary definition (full file):**
- **Population:** `application_type == "Individual"`, `term == " 36 months"`, `issue_d` from 2007-06 to 2015-12.
- **Observation (performance) window:** from the issue month to contractual maturity (36 months). The last cohort (Dec 2015) reaches maturity in Dec 2018, at or before the 2018Q4 snapshot, so **every loan in the population has had its full contractual term to resolve**. This removes right-censoring for the main target.
- **Bad (y = 1):** `loan_status ∈ {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}`.
- **Good (y = 0):** `loan_status ∈ {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}`. Early repayment counts as good.
- **Indeterminate (excluded and counted):** any other status (Current, Late, In Grace) at snapshot. These are loans still open past maturity, e.g. modified or hardship loans. D-03 reports the count per vintage. If it is above 1% of any vintage, D-03 flags that vintage, and a sensitivity analysis treats them as bad (Late) or good (Current).
- **"Does not meet the credit policy" loans** (2007–2010): kept with a flag. Sensitivity run SA-04 drops them.
- **60-month loans:** excluded from core because their maturity cut-off (issued ≤ 2013-12) would leave too few recent vintages. Stretch task S-04 adds them as a separate segment.

**(b) Zenodo fallback definition:** the dataset already restricts to fully paid vs charged off and has no `term`. Restrict to `issue_d ≤ 2013-12`, the latest issue date at which both 36- and 60-month loans are fully matured by Dec 2018. Accept the smaller and older sample, and disclose it.

**(c) Secondary target (stretch S-05):** "bad within 24 months of issue", approximated as charged off with `last_pymnt_d ≤ issue_d + 24 months`. This depends on LendingClub's charge-off timing policy, which is **Unverified**. It is a sensitivity analysis only.

## 5.4 Cleaning rules (all in `src/credit_risk/data/clean.py`, each with a test)

1. Parse `issue_d` and `earliest_cr_line` with format `%b-%Y`. Rows that fail to parse are dropped and counted.
2. Strip `%` from `revol_util` and `int_rate`; strip whitespace from `term`.
3. Drop exact duplicate `id`. Assert there are none after the drop.
4. Drop the rows with an empty `loan_amnt` (known footer artefact rows in LC CSVs, **Medium**, from Phase 1's cleaning reference). Log the count.
5. Range checks with pandera: `loan_amnt` > 0; `fico_mid` in [300, 850]; `annual_inc` ≥ 0; `0 ≤ dti ≤ 100` (else NaN plus a flag); `credit_age_months` ≥ 0 (negative → NaN).
6. Winsorise nothing before the split. Any capping is fitted on train only, inside the pipeline.
7. Save `data/interim/loans_clean.parquet` with a schema hash, plus `reports/data_profile.md` (rows, bad rate and missingness per vintage).

## 5.5 Split design (out-of-time, never random)

All boundaries are on `issue_d`. Fixed in `configs/splits.yaml` and frozen at tag `prereg-v1`.

| Set | Vintages | Use |
|---|---|---|
| Development (train) | 2007-06 → 2013-12 | Fit binning, WOE, models |
| Validation | 2014-01 → 2014-12 | Hyperparameters, calibration fit, imbalance arm choice, cutoff choice |
| **Out-of-time test** | 2015-01 → 2015-12 | Touched **once** for the final numbers (rule enforced by `make final`, which writes a lock file) |
| Monitoring (unlabelled) | 2016-01 → 2018-12, all terms | PSI and characteristic drift only. Outcomes are not mature, so no performance metrics |

**Stability check across vintages:**
- **Rolling-origin backtest** (M-07). Three folds: train ≤2011 → test 2012; train ≤2012 → test 2013; train ≤2013 → test 2014. Each test year is used only to read the fold's metrics. Report AUC, KS and Brier per fold, plus a CI across folds.
- **Per-quarter metrics** within the OOT year (AUC, KS, bad rate, mean predicted PD).
- **Adversarial validation:** a classifier separating train rows from OOT rows. Its AUC measures shift ([Qian et al.](https://arxiv.org/pdf/2112.10078) · **Medium**). Report its top drifting features.

Caveat: early vintages (2007–2009) are small and include the financial crisis. The model is fitted on them but weighted equally. Sensitivity SA-03 trains on 2010+ only.

## 5.6 Leakage checklist (project-specific)

| # | Check | How it is caught |
|---|---|---|
| L1 | Post-origination columns in `X` | T-01: `set(X.columns) ⊆ whitelist`, and no whitelist entry matches the regex `pymnt|rec_|recover|settle|hardship|out_prncp|last_` |
| L2 | LC model outputs (`grade`, `sub_grade`, `int_rate`, `installment`) in `X` | T-01, same test |
| L3 | Binning or WOE fitted on validation or test rows | Pipeline object fitted only inside `fit(train)`. T-03 refits on a shuffled train and asserts the bin edges are unchanged when validation rows are perturbed |
| L4 | Random split anywhere | T-02 asserts `max(train.issue_d) < min(valid.issue_d) < min(test.issue_d)` |
| L5 | Test set used more than once | `make final` writes `results/final.lock` with a hash. The pipeline refuses to evaluate on test if the lock exists and the config hash differs |
| L6 | Target encoded through missingness (e.g. a field only filled after default) | Missing-rate-by-target table per feature in D-05. Flag any feature whose missing rate differs more than 3× between good and bad (**Unverified** threshold, a heuristic) |
| L7 | Date leakage (features that encode issue year) | Adversarial validation (5.5). Late-added columns excluded |
| L8 | `earliest_cr_line` date arithmetic using the snapshot date instead of `issue_d` | T-04 unit test on hand-made rows |
| L9 | Implausible performance | Any model with OOT AUC > 0.80 halts reporting until a leakage audit is written (Phase 1 heuristic, **Unverified**) |

## 5.7 Bias checklist (sampling, selection and measurement)

| # | Bias | Effect | Handling |
|---|---|---|---|
| B1 | **Selection bias / no rejects** | The model learns only from loans LC approved, so PDs for applicants LC would have rejected are extrapolations | Stated limitation. Reject-inference discussion and optional experiment S-06 (kickout measure, [Kozodoi et al.](https://arxiv.org/pdf/1909.06108) · **High**) |
| B2 | **Survivorship within vintages** | Avoided for the main target by the maturity rule (5.3a). Would reappear if 2016+ loans were labelled | Monitoring set is unlabelled |
| B3 | **Regime shift** (crisis, P2P growth, LC policy changes, 2020 platform exit) | OOT performance may not transfer to today | Per-vintage metrics. No claim of current applicability ([Nasdaq](https://www.nasdaq.com/articles/lendingclub-is-ending-its-p2p-lending-platform-now-what-2020-10-08) · **Medium**) |
| B4 | **Self-reported fields** (income, employment) | Measurement error. Verification status differs by group | Keep `verification_status`. Report its IV |
| B5 | **Geographic proxy for race** | Group labels are area-level, not individual (ecological fallacy) | Labelled "proxy" everywhere. Groups are area composition, not people |
| B6 | **Label definition** (charge-off ≠ 90 DPD) | PD level not comparable to Basel PDs | Model card limitation |
| B7 | **ZIP / state as features** | Can act as a proxy for race (disparate treatment risk) | Never model inputs (5.2). Used only to audit |

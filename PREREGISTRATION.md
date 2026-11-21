# Pre-registration

Registered before any model code (task P-01). Kouki commits this file and tags the commit `prereg-v1`; `scripts/check_prereg.py` (run in CI) asserts that the tag comes before the first commit touching `src/credit_risk/models/`.

**Rules after the tag:** nothing above "Amendments" is edited. Any change to a criterion, a split boundary, the target definition or the feature whitelist goes into a dated entry under "Amendments" that names the file and says why. The checker fails if `configs/splits.yaml`, `configs/target.yaml` or `configs/features.yaml` differ from the tag without such an entry.

Source: `phase2/PLAN.md` §2 (approved 2026-11-08), copied verbatim below.

## 1. Success criteria (PLAN §2, verbatim)

Committed as `PREREGISTRATION.md` and tagged `prereg-v1` **before any model code** (task P-01). Changing a criterion afterwards requires a dated amendment section explaining why. Results are never promised: the published honest range on LC application-time data is about 0.66–0.73 AUC ([Sanz-Guerrero & Arroyo](https://arxiv.org/html/2401.16458v3) · High; [Qian et al.](https://arxiv.org/pdf/2112.10078) · Medium). That range is context, not a target.

**Models compared:**
- M0 naive (train bad rate)
- M1 incumbent (LC `sub_grade` as a score, benchmark only)
- **M2 champion** (WOE logistic scorecard)
- M3 LightGBM unconstrained
- **M4 challenger** (monotone LightGBM, calibrated)

**Primary metrics** on the OOT test set (2015 vintages), each with a 95% bootstrap CI (2,000 resamples) plus DeLong for AUC:
- AUC / Gini, KS, Brier, log-loss, calibration slope and intercept.
- Secondary: PR-AUC, approval rate vs bad rate, EL per $100 at fixed approval rates.

| ID | Hypothesis / criterion | Minimum threshold (pre-registered) | Basis |
|---|---|---|---|
| C1 | **Pipeline validity:** the champion learns real signal | M2 OOT AUC 95% CI lower bound > 0.60 | Unverified. Set below the published range so it tests the pipeline, not ambition |
| C2 | **Leakage tripwire** | Any OOT AUC > 0.80 → stop and write a leakage audit before reporting anything | Phase 1 inference · Unverified |
| C3 | **Challenger promotion:** M4 replaces M2 only if **all** hold: (a) ΔAUC 95% CI lower bound > 0; (b) point ΔAUC ≥ 0.01 (minimum effect of interest); (c) ΔBrier upper CI ≤ 0.001; (d) monotonicity tests pass; (e) reason-code fidelity ≥ 80%; (f) the equalised-odds gap is not larger than M2's by more than its CI | Unverified judgement. (b) roughly matches the size of gain papers treat as meaningful (Qian et al. report +0.009) |
| C4 | **Calibration:** OOT calibration slope in [0.8, 1.2] and |mean PD − observed bad rate| ≤ 2 pp | Unverified. A failure means "requires recalibration before use", not that the model is invalid |
| C5 | **Stability:** rolling-origin AUC sd ≤ 0.02 across the 3 folds; no OOT quarter with AUC below (OOT AUC − 0.03); score PSI dev→OOT below its simulated 99% critical value | Unverified thresholds. PSI method from [du Pisanie et al.](https://arxiv.org/pdf/2303.01227) · High |
| C6 | **Imbalance arm:** chosen by lowest validation log-loss among arms not significantly worse on AUC; tie → natural mix | Rule, not a result |
| C7 | **Explanation quality:** fidelity ≥ 80%; Kendall's W of top-10 SHAP ranking across 10 seeds reported, with W < 0.7 flagged (Unverified threshold) | |
| C8 | **Incumbent comparison:** report M2/M4 vs M1 either way. No threshold, because LC's grade may use information that is not in the file | |

**What a null or negative result looks like, and how to present it:**
- **C3 fails** (challenger not better): "Champion retained. Boosting's gain of [Δ] AUC (CI [a, b]) did not meet the pre-registered bar, so the added complexity is not justified." This is a **legitimate model-risk outcome** and a strong interview story (S12).
- **C1 fails:** a pipeline or target bug is likely. Debug it, document the bug and the fix in `research/postmortems/`, and rerun from validation only (the test stays untouched until fixed; the lock file records a reset with its reason).
- **C4 fails:** report it, apply recalibration-in-the-large on validation, and state that production would need periodic recalibration (Reg B requires periodic revalidation, [1002.2(p)](https://www.law.cornell.edu/cfr/text/12/1002.2) · High).
- **C5 fails:** a regime-dependence finding. Show the per-vintage table and discuss it. Do not hide quarters.
- **M1 beats both:** "LendingClub's own grade outranks a model using only the public application fields. This bounds what the public data can do." That is an honest, interesting result.

## 2. Data, target and splits

- **Data:** Kaggle `wordsforthewise/lending-club` v3, `accepted_2007_to_2018Q4.csv.gz`, CC0; SHA-256 recorded in `configs/base.yaml` and checked on every download (`research/data_decision.md`).
- **Target** (`configs/target.yaml`, data plan 5.3a): individual, 36-month loans issued 2007-06 to 2015-12; bad = Charged Off, Default, or the policy-exception "Charged Off"; good = Fully Paid or the policy-exception "Fully Paid"; Current, In Grace Period and Late statuses are indeterminate (excluded and counted). Charge-off over the loan's life, not Basel 90 DPD.
- **Features** (`configs/features.yaml`): 22 application-time features (whitelist, default deny). `grade`, `sub_grade`, `int_rate`, `installment` are benchmark only; `addr_state`, `zip_code` are fairness only. The D-05 audit (`research/feature_audit.md`) marks `term` and `application_type` as constant in this population.
- **Splits** (`configs/splits.yaml`), on issue month, never random: train 2007-06 to 2013-12, validation 2014-01 to 2014-12, out-of-time test 2015-01 to 2015-12 (touched once, by `python -m credit_risk final`), monitoring 2016-01 to 2018-12 (unlabelled). Rolling-origin backtest folds test on 2012, 2013 and 2014 only.

Counts per split, generated by `scripts/fill_prereg.py` from `results/vintage_profile.csv` (counts only: no outcome rates or metrics are registered here):

<!-- generated:prereg:start -->
| Split | Issue months | Labelled loans | Bad loans | Indeterminate (excluded) |
|---|---|---:|---:|---:|
| train | 2007-06 to 2013-12 | 175,426 | 22,157 | 0 |
| valid | 2014-01 to 2014-12 | 162,570 | 22,315 | 0 |
| test (OOT) | 2015-01 to 2015-12 | 282,787 | 42,089 | 147 |

Config files at registration (SHA-256 of the file with LF line endings):

| File | SHA-256 |
|---|---|
| `configs/base.yaml` | `17581c397a816973484753888aa3e834e0bea9ead22b19867e0a0952bb123712` |
| `configs/business.yaml` | `7c5673e1c123d18d4475c4b32bf1f22868d66b06cd2e4ab45d9387fe0756690c` |
| `configs/features.yaml` | `b01067a9213e8ea84f44a71e362e44c9d8da4be2941e866008d09f1b2204da92` |
| `configs/monotone.yaml` | `410dd9eeb451a57e3ef22bd636b5fe78f9e77159abdc02a5ff26c5b8c920edd9` |
| `configs/reason_text.yaml` | `a7222f04f7e1e5a3d9d89fcecbc462647c253e6fb1216eaec6365d364b033e6b` |
| `configs/scorecard.yaml` | `61099a2c98264dad4b6f11afd3feb50368c146d97dedffd004c98368663c02ba` |
| `configs/splits.yaml` | `966c4cbbedaeefb2338af0211a905ed579150536027be2523bf3ff1fedeb1966` |
| `configs/target.yaml` | `190ea1c37b885db1131a10aca20affb3dd69e2c633053392efdcdaab0de09a8a` |
<!-- generated:prereg:end -->

## Amendments

None.

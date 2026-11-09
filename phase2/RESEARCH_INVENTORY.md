# Research inventory (Step 0)

Written 2026-11-06. This file records what research already existed before Phase 2 planning, what it found, what conflicts or gaps it has, and what I re-verified today.

Confidence tags follow Phase 1: **High** = primary source opened and read; **Medium** = reputable secondary source, or a primary source only partly readable; **Low** = blog or single anecdote; **Unverified** = judgement with no source.

## 1. What is in the folder

| Location | Type | Contents |
|---|---|---|
| `/mnt/project-files/` (shared project folder) | Folder | **Empty** when Phase 2 started. No notes, datasets, drafts or code. |
| Linked computer | Device | No folders shared with this thread, so nothing to read there. |
| [Credit Default Prediction: Phase 1 Research](https://claude.ai/artifact/QNuEAEiJEPbSVP9VmvTWMq) | Claude Doc (the only prior output) | 12-section research report, 53 sources, all marked as opened on 2026-11-06. Read in full for this inventory. |
| Phase 1 thread ("Research credit default prediction project") | Chat history | The Phase 1 brief and a 4-bullet summary. It adds no facts beyond the report. |

So the research base is one document. It is substantial, so I did **not** redo the research. I re-verified only the items that are load-bearing for plan decisions (section 4 below).

## 2. Key findings carried over from Phase 1 (with their sources)

| # | Finding | Source · confidence |
|---|---|---|
| F1 | A CC BY 4.0 Lending Club extract exists: 1,347,681 loans, 2007–2018, application-time variables only, target = charged off vs fully paid. | [Zenodo 11295916](https://zenodo.org/records/11295916) · **High** (re-verified, see V1) |
| F2 | Honest application-time AUC on Lending Club is about 0.66–0.73. 0.6575→0.6644 (XGBoost, Zenodo extract, cross-validated); 0.724→0.733 out-of-time on 2018–2020. | [Sanz-Guerrero & Arroyo](https://arxiv.org/html/2401.16458v3) · **High**; [Qian et al.](https://arxiv.org/pdf/2112.10078) · **Medium** |
| F3 | An AUC above about 0.80 on this data should trigger a leakage review. | Phase 1 inference from F2 · **Unverified** |
| F4 | Post-origination fields (total_pymnt, recoveries, last_pymnt_d, out_prncp, settlement and hardship flags) leak the outcome. | [Crumley](https://daniellecrumley.github.io/LendingClubProject/Data-Cleaning.html) · **Medium** |
| F5 | Resampling (ROS/RUS/SMOTE) distorts probabilities and does not improve AUC. Threshold moving on a natural class mix is preferred. | [van den Goorbergh et al. 2022](https://arxiv.org/abs/2202.09101v1) · **High**; [Elor & Averbuch-Elor 2022](https://arxiv.org/abs/2201.08528v3) · **High** |
| F6 | LightGBM `is_unbalance` / `scale_pos_weight` give "poor estimates of the individual class probabilities". | [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) · **High** (re-verified) |
| F7 | Ensembles and boosting usually beat logistic regression on AUC; logistic regression is competitive on Brier. | [Lessmann et al. 2015](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) · **High** |
| F8 | Some explainers were no better than random features at finding adverse-action drivers, so explanations need a fidelity test. | [FinRegLab 2023](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) · **High** |
| F9 | Mid-ranked SHAP features are unstable across seeds; cost-sensitive models give less stable explanations. | [Lin & Wang 2025](https://arxiv.org/pdf/2508.01851) · **Medium**; [Ballegeer et al. 2025](https://ideas.repec.org/a/eee/ejores/v326y2025i3p630-640.html) · **High** |
| F10 | Fairness: separation (equalised odds) is the recommended criterion for scorecards; less discriminatory alternatives (LDAs) are usually findable. | [Kozodoi et al.](https://arxiv.org/abs/2103.01907v3) · **High**; [Gillis et al. 2024](https://facctconference.org/static/papers24/facct24-27.pdf) · **High** |
| F11 | Lending Club has no race or sex field; BISG needs surname and geocode, which LC lacks. | [CFPB proxy methodology](https://www.github.com/cfpb/proxy-methodology) · **High** |
| F12 | SR 26-2 (17 Apr 2026) replaced SR 11-7. | [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) · **High** (re-verified, see V4) |
| F13 | CFPB Circulars 2022-03 and 2023-03 were withdrawn on 12 May 2025; the Reg B duty to give specific reasons stands. | [CFPB withdrawn guidance](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/) · **High** |
| F14 | EU AI Act: obligations for Annex III high-risk systems (credit assessment) moved to 2 Dec 2027 by Regulation (EU) 2026/1744, in force 27 July 2026. | [Mondaq](https://www.mondaq.com/eu-digital-omnibus-on-ai-enters-into-force/1825722) · **Medium** (re-verified, see V6) |
| F15 | Free hosting: Streamlit Community Cloud gives 0.078–2 cores and 690 MB–2.7 GB RAM, and apps sleep after 12 hours idle. On Hugging Face, Gradio and Docker Spaces need a paid plan to create. | [Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) · **High** (re-verified); [HF Spaces](https://huggingface.co/docs/hub/en/spaces-overview) · **High** |
| F16 | Phase 1 roadmap items 1–10 (about 10 h) are pre-build checks; their outputs feed this plan's Phase 0. | Phase 1 §8 |

## 3. Contradictions and errors found

| # | Issue | Resolution |
|---|---|---|
| C1 | **Phase 1 says the Zenodo extract is enough for the main model, but it has only 14 columns.** Re-verification shows: id, issue_d, revenue, dti_n, loan_amnt, fico_n, experience_c, emp_length, purpose, home_ownership_n, addr_state, zip_code, title, desc. It has **no `term`, no payment dates, no delinquency, inquiry or revolving-utilisation history**. | Phase 1 understated this. With about 10 usable predictors, a WOE scorecard is thin, and the observation window cannot be defined precisely without `term`. The plan makes the **full Lending Club accepted-loans file the primary dataset** (behind a licence-and-access gate) and keeps Zenodo as the licensed fallback and a reproduction benchmark. See docs/02_data_plan.md. |
| C2 | Phase 1 proposes a fairness audit on **German Credit by sex**. The UCI page for the corrected "South German Credit" says that in the original data "sex cannot be recovered from the variable, because male singles and female non-singles are coded with the same code (2)" ([UCI 573](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) · **High**). | Use **South German Credit** (CC BY 4.0) and **age** as the protected attribute. Age is a prohibited basis under Reg B ([12 CFR 1002.2(z)](https://www.law.cornell.edu/cfr/text/12/1002.2) · **High**). Drop the sex audit. |
| C3 | Phase 1's legal framing assumed disparate impact applies under ECOA. **The CFPB amended Regulation B with a final rule published 22 Apr 2026 and effective 21 Jul 2026. It says "ECOA does not authorize disparate-impact liability" and removes the effects test.** The rule is being challenged in D.D.C. (filed 27 May 2026, no preliminary injunction sought, briefing runs to 12 Jan 2027). | Fairness section re-framed: (a) the federal ECOA effects test is out as of now, but **state laws, the Fair Housing Act (for housing-secured credit), private plaintiffs, CRA and disparate treatment via proxies remain**; (b) the legal position is in litigation, so the README must date its claims. Sources: [Venable](https://www.venable.com/insights/publications/2026/05/cfpb-makes-significant-changes-to-regulation-b) · **Medium**, [Husch Blackwell](https://www.huschblackwell.com/newsandinsights/cfpb-finalizes-major-regulation-b-overhaul-disparate-impact-out-discouragement-narrowed-and-spcps-restricted) · **Medium**, [Consumer Finance Monitor](https://www.consumerfinancemonitor.com/2026/08/10/lawsuit-challenging-cfpbs-revised-ecoa-rule-moves-forward-under-lengthy-summary-judgment-schedule/) · **Medium**. I could not open the Federal Register page itself (site error), so these are law-firm summaries. Two firms agree on the key facts. |
| C4 | Phase 1 cites two different honest AUC ranges (0.66 on Zenodo CV vs 0.72–0.73 out-of-time on 2018–2020). | Both are kept. They differ in features and years. With the richer full-file features the plan expects the top of that range, but **no result is promised**. |
| C5 | Phase 1 assumed 4 weeks at 10 h/week and intermediate skill. The Phase 2 brief says "no time limit, no skill limit". | The 40-hour cap is dropped. Former stretch items (rolling-origin backtests, FastAPI, reject-inference discussion, LDA search) move into core where the brief requires them. Learning paths cover weak areas. |
| C6 | Phase 1 listed FastAPI as out of scope. The Phase 2 brief requires a scoring API plus an app. | FastAPI is now core. |
| C7 | Phase 1 relied on OptBinning for points-to-double-odds scaling. I could open only OptBinning's tutorial and GitHub README, and both show only `min_max` scaling. Its API docs site errored. | The plan implements PDO scaling itself as a ~15-line tested function (`scorecard.scale_points`). That makes the formula easy to defend and does not depend on unverified library behaviour. |

## 4. Re-verification done today (2026-11-06)

| # | Item | Result | Source · confidence |
|---|---|---|---|
| V1 | Zenodo LC extract: access, licence, schema | Open download, CC BY 4.0, v0.1 (25 May 2024), 167.5 MB, md5 `b019384d6bc65bf2a3e839362e4ff502`. Target: 0 = fully paid, 1 = charged off. Current, In Grace and Late loans excluded. 14 columns (see C1). grade and int_rate deliberately excluded. | [Zenodo](https://zenodo.org/records/11295916) · **High** |
| V2 | Full LC accepted file (Kaggle `wordsforthewise/lending-club`) | **Could not open from my environment**: kaggle.com is blocked by the network policy, and the page returned only metadata (title "All Lending Club loan data"). A secondary catalogue says it is a "CC0-tagged snapshot", ~2.26M rows × 151 columns accepted and ~27M × 9 rejected. A data dictionary that matches this Kaggle version is on figshare (no licence shown). | [Credit-score book catalogue](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html) · **Medium**; [figshare dictionary](https://figshare.com/articles/dataset/Lending_club_dataset_description/20016077) · **Medium**. **Licence still Unverified → task D-01 checks it on your machine.** |
| V3 | Library versions (PyPI JSON API) | lightgbm 4.7.0 · shap 0.54.1 (Python ≥3.12) · optbinning 1.0.0 · fairlearn 0.15.0 · scikit-learn 1.9.1 · xgboost 3.4.1 · catboost 1.2.10 · pandas 3.1.1 · numpy 2.5.4 (Python ≥3.12) · scipy 1.18.1 · streamlit 1.67.0 · fastapi 0.162.0 · pydantic 2.14.3 · pandera 0.35.0 · imbalanced-learn 0.14.2 · uv 0.13.7 | pypi.org JSON API, queried today · **High**. Whether these versions work together (e.g. optbinning with pandas 3) is **Unverified**; task E-01 checks it. |
| V4 | SR 26-2 | Issued 17 Apr 2026 by the Fed, OCC and FDIC. Supersedes SR 11-7 and SR 21-8. "Most relevant" to banks above $30B in assets. Risk-based by materiality. Validation covers conceptual soundness (including "benchmarking to other models" and "interpretability measures"), outcomes analysis and back-testing, and ongoing monitoring. Generative and agentic AI are out of scope (footnote 3). "Challenger model" is not named. The closest wording is "comparison of alternative assumptions and methodologies". | [SR 26-2 letter](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm), [attachment](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) · **High** |
| V5 | Reg B reasons and definitions | No single method is required. Two approved methods: compare the applicant's factor scores with the average of applicants at or just above the passing score, or with the average of all applicants. More than four reasons is "not likely to be helpful". Reasons "must relate to and accurately describe the factors actually considered or scored". Elderly = 62+. A scoring system must be "periodically revalidated". | [Interp. §9](https://www.consumerfinance.gov/rules-policy/regulations/1002/2026-07-21/interp-9/), [12 CFR 1002.2](https://www.law.cornell.edu/cfr/text/12/1002.2) · **High** |
| V6 | EU AI Act dates | Regulation (EU) 2026/1744 entered into force 27 Jul 2026. Annex III dates move from 2 Aug 2026 to 2 Dec 2027. "Credit assessment" is named among the deferred areas. | [Mondaq](https://www.mondaq.com/eu-digital-omnibus-on-ai-enters-into-force/1825722) · **Medium** |
| V7 | Free hosting for the API | Render free web service: 512 MB RAM, spins down after 15 min idle, spin-up "takes about one minute", 750 free instance hours/month, ephemeral disk. | [Render free docs](https://render.com/docs/free), [pricing](https://render.com/pricing) · **High** |
| V8 | CI | GitHub Actions on standard hosted runners is free for public repositories. | [GitHub billing docs](https://docs.github.com/en/billing/concepts/product-billing/github-actions) · **High** |
| V9 | PSI thresholds | The 0.1 / 0.25 rules of thumb have no statistical justification. Critical values depend on sample size and number of bins. Prefer asymptotic or simulated critical values. | [du Pisanie et al. 2023](https://arxiv.org/pdf/2303.01227) · **High** |
| V10 | SHAP scale | `model_output="probability"` needs `feature_perturbation="interventional"` with 100–1,000 background rows. The default `raw` output is log-odds for binary boosting. | [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) · **High** |
| V11 | Calibration | Isotonic regression is as good or better than sigmoid with more than ~1,000 samples. Sigmoid keeps the ranking; isotonic can create ties that change AUC. `FrozenEstimator` is used to calibrate a model that is already fitted. Docs are for 1.9.1. | [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) · **High** |
| V12 | Reject inference | Gains are modest. Evaluating on accepted loans only is misleading: the rank correlation with performance on an unbiased sample was about −0.0009. The paper proposes a "kickout" measure. | [Kozodoi et al., Shallow self-learning](https://arxiv.org/pdf/1909.06108) · **High** |

## 5. Gaps that remain

1. **Kaggle licence and access** for the full LC file (V2). Blocks the primary-data choice. Task D-01 resolves it on your machine; the fallback is defined.
2. **Exact row counts per vintage and term** in the full file. Not knowable until download (task D-02). The plan uses no invented counts.
3. **Census join for a geographic fairness proxy.** Not verified (Census API access, ZIP3-to-ZCTA mapping). Task F-01 is a spike with a written fallback.
4. **Charge-off timing policy at Lending Club** (how many days past due before charge-off). Unverified. It matters only for the secondary 24-month target, which is stretch.
5. **Library compatibility** (optbinning 1.0.0 with pandas 3.1 / numpy 2.5). Unverified. Task E-01.
6. **Interview evidence for asset-manager roles.** The Goldman Sachs asset-management quant posting I opened lists no skills, so the asset-manager signals rest on the other firms' postings (Medium).

## 6. Claims still Unverified (carried into the plan with that label)

- An AUC above 0.80 means leakage (F3).
- The challenger should beat the champion by a few AUC points, not tens.
- A minimum effect of interest of ΔAUC ≥ 0.01. This is a pre-registered judgement, not a fact.
- Profit and loss assumptions for the business view (revenue per good loan). They are parameters with sensitivity analysis.
- A laptop with 16 GB RAM handles the full file in LightGBM in minutes. Task E-01 measures this.

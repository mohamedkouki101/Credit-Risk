# Credit Default Prediction: Phase 1 Research

Nov 6, 2026 · @Kouki

## Assumptions

This is the Phase 1 research only: no code and no execution plan. The brief left some placeholders, so I filled them as follows.

- **Skill level:** intermediate Python and ML (pandas, scikit-learn, basic model evaluation).
- **Time budget:** 4 weeks at 10 hours per week, about 40 hours in total.
- **Tools line:** Lending Club or a similar public dataset, logistic regression vs gradient boosting, SHAP for explainability.
- **Show line:** class imbalance handling, AUC and KS, a points-based scorecard, model fairness and interpretability, framed for banks.
- **Today:** 2026-11-06. Sources older than three years are flagged when recency matters.

**Confidence tags.** High = primary source (paper, official docs, regulator, repo) I opened and read. Medium = reputable secondary source or a primary source I only skimmed. Low = blog, forum or single anecdote. Unverified = my own professional judgement with no source; treat as a hypothesis to test.

## 1. Problem and context

The task is to estimate each applicant's probability of default (PD), turn it into an accept/reject or pricing decision, and explain that decision to the borrower, the regulator and the model validator. That is the same problem a bank's retail credit team solves for every card, personal loan or mortgage.

**What "default" means in a bank.** Under the Basel IRB rules a default is either "unlikely to pay" or "past due more than 90 days on any material credit obligation" (up to 180 days allowed for retail by some supervisors). Retail PD models must use at least five years of loss history, and realised default rates must be back-tested against predicted PDs per grade at least annually. [BCBS CRE36](https://www.bis.org/committees/bcbs/basel-framework/81373/chapter.pdf) · **High**

**Who uses it.**

| User | What they do with a PD model | Source · confidence |
| --- | --- | --- |
| Retail lenders (underwriting) | Approve or decline, set limit and rate from a score; FICO says its score is used by 90% of top US lenders (vendor claim, no methodology) | [myFICO](https://www.myfico.com/credit-education/credit-scores/how-lenders-use-credit-scores) · **Medium** |
| Capital (Basel IRB banks) | PD feeds risk-weighted assets and regulatory capital | [BCBS CRE36](https://www.bis.org/committees/bcbs/basel-framework/81373/chapter.pdf) · **High** |
| Accounting provisions (IFRS 9) | A forward-looking expected credit loss model replaced IAS 39's incurred-loss model in 2014 | [EY on IFRS 9](https://ey.com/en_gl/ifrs-technical-resources/impairment-of-financial-instruments-under-ifrs-9) · **Medium** |
| Model risk and validation teams | Check conceptual soundness, outcomes analysis and ongoing monitoring; US guidance was revised as SR 26-2 on 17 April 2026, replacing SR 11-7 | [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) · **High** |
| Compliance and fair lending (US) | Regulation B requires a statement of specific principal reasons for a decline; "failed to achieve a qualifying score" is not enough, and more than four reasons is "not likely to be helpful" | [12 CFR 1002.9](https://www.law.cornell.edu/cfr/text/12/1002.9), [Official interpretation](https://www.consumerfinance.gov/rules-policy/regulations/1002/2026-07-21/interp-9/) · **High** |

**Why it matters now.** Three pressures meet in this project. Gradient boosting usually ranks borrowers better than a logistic scorecard ([Lessmann et al. 2015](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) · **High**). But the law still requires reasons a borrower can act on ([12 CFR 1002.9](https://www.law.cornell.edu/cfr/text/12/1002.9) · **High**). Note the source conflict: CFPB Circulars 2022-03 and 2023-03, which said complex algorithms are no excuse, were **withdrawn on 12 May 2025** ([CFPB withdrawn guidance list](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/) · **High**). I trust the CFPB's own list over older law-firm summaries; the regulation itself was not withdrawn, so the reason-code requirement stands. And flexible models can widen gaps between groups: in US mortgages, Black and Hispanic borrowers were less likely to gain from ML ([Fuster et al., Journal of Finance 2022](https://www.sfi.ch/de/publications/predictably-unequal-the-effects-of-machine-learning-on-credit-markets.) · **High**). In the EU, credit scoring is a high-risk AI use; the AI Omnibus (Regulation (EU) 2026/1744, in force 27 July 2026) moved those obligations from 2 August 2026 to 2 December 2027 ([Mondaq](https://www.mondaq.com/eu-digital-omnibus-on-ai-enters-into-force/1825722) · **Medium**; agreement details in [Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/) · **Medium**).

**Why Lending Club data.** It is the best-known public loan-level book with real outcomes (2007 to 2018), but it is a closed history. LendingClub retired its retail notes platform on 31 December 2020 after agreeing to buy Radius Bank ([Nasdaq / Motley Fool](https://www.nasdaq.com/articles/lendingclub-is-ending-its-p2p-lending-platform-now-what-2020-10-08) · **Medium**), so no fresh data will arrive. Treat it as a historical case study, not a live book.

## 2. Existing projects and solutions

The published ceiling on Lending Club application-time data is modest: honest out-of-time AUCs sit around 0.66 to 0.73, and any portfolio project reporting 0.90+ on this data almost certainly leaks post-origination fields (my inference from the rows below, **Medium**). Strong gradient boosting plus feature work is the consistent winner; fairness and regulator-grade explanation are where most public work stops.

| Name | Approach | Data | Results | Limitations | Confidence |
| --- | --- | --- | --- | --- | --- |
| [Lessmann, Baesens, Seow & Thomas (EJOR 2015)](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) | Benchmark of 41 classifiers (1,141 models) | 7 retail credit datasets | Ensembles best AUC on 6 of 7; random forest best single method; logistic regression good on Brier score, weak on AUC | Pre-2015 data; no fairness or explainability; does not cover LightGBM/XGBoost era | High |
| [Kozodoi, Jacob & Lessmann (EJOR 2022)](https://arxiv.org/abs/2103.01907v3) | Compares fairness criteria and pre/in/post-processors on profit-oriented scorecards | Real credit datasets (not named on abstract page) | Recommends separation (equalised odds); fair in-processors cut discrimination at small profit cost | Group labels needed; US legal limits on using them not resolved | High |
| [Fuster et al., "Predictably Unequal?" (J. Finance 2022)](https://www.sfi.ch/de/publications/predictably-unequal-the-effects-of-machine-learning-on-credit-markets.) | Traditional vs ML default models plus an equilibrium credit market model | US mortgages | ML gains skew away from Black and Hispanic borrowers; flexibility widens rate gaps | Mortgage data with race; not directly replicable on Lending Club | High |
| [FinRegLab, Blattner & Spiess (2023)](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) | Tests SHAP, LIME, permutation importance and 7 vendor tools on fidelity for adverse action and fairness | Real credit-bureau-based underwriting data | Some tools no better than random features; dropping "disparity driver" features did little; joint optimisation and adversarial debiasing gave efficient fairness/accuracy trade-offs | Vendor names masked; models not production grade | High |
| [Gillis, Meursault & Ustun (FAccT 2024)](https://facctconference.org/static/papers24/facct24-27.pdf) | Mixed-integer search for less discriminatory alternatives (LDAs) under a business-necessity test | FICO, German credit, income | LDAs found in all audits; about 15 of 18 gains held on test data | Linear models only; refutation only within the model class | High |
| [Namvar et al. (≈2018)](https://arxiv.org/pdf/1805.00801) | Logistic, LDA, random forest with RUS, ROS, SMOTE, ADASYN, hybrids | Lending Club 2016–2017, 66k loans after leakage removal | RF with random undersampling best G-mean 0.65 | Optimises G-mean, ignores calibration; small slice of years | Medium |
| [Qian et al. (arXiv 2021)](https://arxiv.org/pdf/2112.10078) | Adversarial validation to handle dataset shift | Lending Club 2018–2020, out-of-time test | Test AUC 0.724 baseline to 0.733 best; adversarial AUC 0.968 shows strong shift | Gains about 0.01 AUC; one split | Medium |
| [Sanz-Guerrero & Arroyo (arXiv, v3 2025)](https://arxiv.org/html/2401.16458v3) | Fine-tuned BERT on loan descriptions as an extra XGBoost feature | Lending Club, 1.35M loans; 119k with text | AUC 0.6575 to 0.6644 (DeLong significant) | Approved loans only; text only 2008–2014; black-box feature | High |
| [deepsense.ai, Home Credit (Kaggle 2018)](https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/) | 10,000+ hand features, bagged LightGBM, nested single-source models, small CNN/LSTM | Home Credit, \~300k applicants, 7 tables | \~0.80 AUC; 5th private leaderboard | Competition-grade complexity no bank would validate; no fairness work | Medium |
| [Feng et al., CALM (arXiv 2023/24)](https://arxiv.org/abs/2310.00566v3) | Instruction-tuned LLM for credit scoring | 9 datasets, \~14k samples | Claims parity with or gains over classic models | Preprint; small data; flags LLM bias | Medium |
| [gegznav Lending Club project](https://gegznav.github.io/ds-projects/lending-club) | LightGBM, RF, NB, SGD; GCP API | LC 2018 accepted + rejected | No metrics reported on page | One year only; acceptance not default target | Low |
| [OptBinning](https://libraries.io/pypi/optbinning) (tool) | Optimal monotonic binning, WoE/IV, points scorecard | Library, Apache-2.0, v1.0.0 Sep 2026 | Standard open-source scorecard builder | No built-in fairness | High |

## 3. Limits and common pitfalls

The three mistakes that most often sink Lending Club projects are post-origination leakage, random (not time-based) splits, and resampling that wrecks probability calibration. Each has a cheap guard.

| Pitfall | Type | What goes wrong | Guard | Source · confidence |
| --- | --- | --- | --- | --- |
| Post-origination leakage | Technical | Fields like `total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`, settlement and hardship flags are only known after the loan runs; they make AUC look near-perfect | Whitelist features known at application; a reference list exists | [Crumley LC cleaning notes](https://daniellecrumley.github.io/LendingClubProject/Data-Cleaning.html) · **Medium**; application-only version in [Zenodo dataset](https://zenodo.org/records/11295916) · **High** |
| Using LC's own `grade`, `sub_grade`, `int_rate` | Methodological | These are outputs of LendingClub's own risk model; using them means you are partly re-learning their score | Report models with and without them; treat as a "challenger vs incumbent" comparison | **Unverified** (my judgement) |
| Look-ahead in validation | Methodological | Random K-fold mixes 2018 loans into training for 2015 tests; offline AUC overstates live AUC | Out-of-time split by `issue_d`; adversarial validation showed train/test AUC 0.968 separability on LC 2018–2020 | [Qian et al.](https://arxiv.org/pdf/2112.10078) · **Medium** |
| Immature loans (right censoring) | Methodological | Loans still "Current" at the data snapshot are dropped, so recent vintages over-represent early defaulters or early payoffs | Restrict to vintages old enough to mature, or model time-to-default | [Survival analysis on LC](https://systemsforecasting.com/2016/07/application-of-survival-analysis-to-p2p-lending-club-loans-data/) · **Low**; censored spells in [Botha et al. 2026](https://arxiv.org/pdf/2603.11897) · **High** |
| Selection bias (no rejects) | Finance-specific | Outcomes exist only for approved loans, so the model never learns about the people LC turned down | State it as a limitation; optional reject-inference experiment on the 27.6M rejected file | [Sanz-Guerrero & Arroyo](https://arxiv.org/html/2401.16458v3) · **High**; rejected file size in [gegznav](https://gegznav.github.io/ds-projects/lending-club) · **Low** |
| Resampling wrecks calibration | Technical | Random over/undersampling and SMOTE strongly overestimate minority-class probability and do not improve AUC | Train on natural class mix; move the decision threshold instead; recalibrate if you do weight | [van den Goorbergh et al. 2022](https://arxiv.org/abs/2202.09101v1) · **High**; [Elor & Averbuch-Elor 2022](https://arxiv.org/abs/2201.08528v3) · **High** |
| Class weights hurt probabilities | Technical | LightGBM's `is_unbalance` / `scale_pos_weight` make individual class probabilities less reliable | Prefer unweighted training + calibration, or weight then recalibrate | [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.4.0/Parameters.html) · **High** |
| Single-metric reporting | Methodological | AUC implicitly uses different cost assumptions per classifier; it says nothing about calibration | Report AUC, KS, Brier, calibration plot, and an expected-loss or profit view | [Lessmann et al. 2015](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) · **High** |
| SHAP misuse | Technical | Default TreeSHAP explains log-odds, not probability; probability output needs interventional mode with background data | Decide the explained scale up front; use 100–1,000 background rows | [SHAP TreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) · **High** |
| Explanations that do not reflect the model | Regulatory | Some explanation tools were no better than random features at finding adverse action drivers | Check explanation fidelity (e.g. perturb top reasons and confirm PD moves) | [FinRegLab 2023](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) · **High** |
| Fairness without protected attributes | Finance-specific | LC has no race or sex field, so group fairness cannot be measured directly; the regulator's workaround (BISG) needs surname and geocode, which LC lacks | Use a geography-only proxy (e.g. state or 3-digit ZIP joined to Census data) and label it a proxy; or run the fairness module on German Credit, which has sex and age | [CFPB proxy methodology](https://www.github.com/cfpb/proxy-methodology) · **High**; [UCI German Credit](https://archive.ics.uci.edu/dataset/144) · **High** |
| Dropping "bias" features as a fix | Finance-specific | Removing a few disparity-driving features barely improved fairness and cut accuracy | Use in-processing (e.g. reductions) or a systematic LDA search | [FinRegLab 2023](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) · **High**; [Gillis et al. 2024](https://facctconference.org/static/papers24/facct24-27.pdf) · **High** |
| Model risk and governance | Regulatory | A bank validator expects conceptual soundness, outcomes analysis and ongoing monitoring; a notebook alone is not validatable | Ship a model card / validation report structured on those three pillars | [SR 26-2 guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) · **High** |
| Regime shift | Finance-specific | 2007–2018 spans the financial crisis and a P2P boom; LC stopped retail notes in 2020 | Report AUC per vintage year; do not claim the model applies to today's borrowers | [Nasdaq](https://www.nasdaq.com/articles/lendingclub-is-ending-its-p2p-lending-platform-now-what-2020-10-08) · **Medium** |

## 4. Research frontier (2023 to 2026)

The field is moving on five fronts: tabular foundation models, explanation reliability, systematic fairness search, text and LLM signals, and a reshaped regulatory map. Gradient boosting is still the practical default for a 1M-row loan book.

| Trend | What is new | Implication for this project | Source · confidence |
| --- | --- | --- | --- |
| Tabular foundation models | TabPFN (Nature, Jan 2025) beats tuned boosting on datasets up to 10,000 samples | Too small a limit for full LC; could be a side experiment on a 10k sample or on German Credit | [TabPFN, Nature 2025 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/39780007/) · **High** |
| Trees vs deep learning, re-benchmarked | TabArena (NeurIPS 2025): CatBoost ranks first without ensembling; tuned deep nets catch up only with post-hoc ensembling; cross-model ensembles are state of the art | Keep LightGBM/XGBoost/CatBoost as challenger; deep learning is not worth the 40-hour budget | [TabArena](https://arxiv.org/pdf/2506.16791) · **High**; earlier [Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815) · **High** |
| Explanation stability | SHAP top and bottom features are stable across 100 seeds, mid-ranked ones are not (2025); cost-sensitive models give markedly less stable explanations, worse with more imbalance (EJOR 2025) | Report SHAP rank stability across seeds before using SHAP for reason codes | [Lin & Wang 2025](https://arxiv.org/pdf/2508.01851) · **Medium**; [Ballegeer et al., EJOR 2025](https://ideas.repec.org/a/eee/ejores/v326y2025i3p630-640.html) · **High** |
| Explanation fidelity | Open-source and vendor explainers vary from strong to no better than random at finding true decline drivers | Add a fidelity check, not just a SHAP plot | [FinRegLab 2023](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) · **High** |
| Less discriminatory alternatives (LDA) | Formal search for models with equal error rates and smaller group gaps (FAccT 2024); model multiplicity makes LDAs common | Show a small LDA-style search: several near-equal models, pick the fairest | [Gillis, Meursault & Ustun 2024](https://facctconference.org/static/papers24/facct24-27.pdf) · **High** |
| Text and LLM signals | BERT on loan descriptions adds \~0.007 AUC on LC; instruction-tuned LLMs (CALM) claim parity on small datasets but carry bias | Optional stretch only; low payoff per hour | [Sanz-Guerrero & Arroyo](https://arxiv.org/html/2401.16458v3) · **High**; [Feng et al.](https://arxiv.org/abs/2310.00566v3) · **Medium** |
| Time-aware modelling | Survival models beat cross-sectional logistic regression on time-dependent discrimination for IFRS 9 term structures (2026); adversarial validation for shift | Time-to-default or vintage curves would set the project apart | [Botha et al. 2026](https://arxiv.org/pdf/2603.11897) · **High**; [Qian et al.](https://arxiv.org/pdf/2112.10078) · **Medium** |
| Regulation, US | SR 26-2 (17 Apr 2026) replaces SR 11-7, tiered by materiality, mentions interpretability as a validation tool, excludes generative/agentic AI; CFPB withdrew its algorithmic adverse-action circulars on 12 May 2025 but Reg B reason rules remain | Frame the model card on SR 26-2; build reason codes to the Reg B standard | [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) · **High**; [CFPB withdrawn list](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/) · **High** |
| Regulation, EU | Credit scoring is high-risk under the AI Act; Annex III duties delayed to 2 Dec 2027 by Regulation (EU) 2026/1744 | Mention EU readiness (documentation, human oversight) in README | [Mondaq](https://www.mondaq.com/eu-digital-omnibus-on-ai-enters-into-force/1825722) · **Medium** |

## 5. Data and tools

Use the CC BY 4.0 application-time Lending Club extract from Zenodo as the main dataset, and German Credit as a small fairness side-dataset with real sex and age fields. Everything runs on a laptop; no GPU is needed.

### Datasets

| Dataset | Size | Licence | Access | Fit for this project | Source · confidence |
| --- | --- | --- | --- | --- | --- |
| **Lending Club granting-model extract (recommended main)** | 1,347,681 loans, 167.5 MB CSV, 2007–2018, binary default vs fully paid, application-time variables only | CC BY 4.0 | Direct download, DOI 10.5281/zenodo.11295916 | Leakage already removed; citable; matches published papers' AUCs | [Zenodo](https://zenodo.org/records/11295916) · **High** |
| Lending Club full accepted + rejected | \~2.26M × 151 accepted; \~27M × 9 rejected | Kaggle mirror tagged CC0; original LC release page gone since 2020 | Kaggle community mirror | Needed only for reject inference or your own feature choices | [Credit-score book catalog](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html) · **Medium**; [gegznav](https://gegznav.github.io/ds-projects/lending-club) · **Low** |
| UCI German Credit (fairness side-dataset) | 1,000 × 20; cost matrix (bad-as-good = 5, good-as-bad = 1) | CC BY 4.0 | UCI repository | Has personal status/sex, age, foreign worker; used in LDA papers | [UCI](https://archive.ics.uci.edu/dataset/144) · **High** |
| Home Credit Default Risk | 307,511 × 122 main table, \~2.5 GB, 7–8 tables | Kaggle competition rules; no raw redistribution | Kaggle account | Richer behaviour data; heavier feature engineering | [Catalog](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html) · **Medium**; [deepsense](https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/) · **Medium** |
| Freddie Mac Single-Family Loan-Level | \~56M mortgages 1999–Jun 2026, origination + monthly performance | Free for non-commercial research; registration | Clarity portal login | Best for time-to-default and vintages; big | [Freddie Mac](https://freddiemac.com/research/datasets/sf-loanlevel-dataset) · **High** |
| Give Me Some Credit | 150,000 × 11, 6.68% default | Kaggle rules, no redistribution | Kaggle | Simple sanity benchmark only | [Catalog](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html) · **Medium** |
| HMDA | \~16M applications (2022) × 99, has race/ethnicity/sex | US public record | CFPB / FFIEC | Approval-decision fairness only; no default outcome | [Catalog](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html) · **Medium** |

Source conflict: the credit-score book calls German Credit "public domain"; the UCI page says CC BY 4.0. I trust UCI, the publisher.

### Libraries

| Need | Library | Note | Source · confidence |
| --- | --- | --- | --- |
| Scorecard (binning, WoE/IV, points) | OptBinning 1.0.0 (Apache-2.0) | Monotonic binning, `Scorecard` class, counterfactuals | [libraries.io](https://libraries.io/pypi/optbinning) · **High** |
| Gradient boosting | LightGBM (or XGBoost / CatBoost) | `monotone_constraints` for business-sensible directions | [LightGBM docs](https://lightgbm.readthedocs.io/en/v4.4.0/Parameters.html) · **High** |
| Explainability | SHAP `TreeExplainer` | Choose log-odds vs probability scale explicitly | [SHAP docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) · **High** |
| Fairness metrics and mitigation | Fairlearn (`MetricFrame`, `ExponentiatedGradient`, `ThresholdOptimizer`) | API docs at 0.16 dev | [Fairlearn API](https://fairlearn.org/main/api_reference/index.html) · **High** |
| KS statistic | SciPy `ks_2samp` (max gap between the two score CDFs) | Or compute directly from ROC (max TPR − FPR) | [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) · **High** |
| Baseline, calibration, splits | scikit-learn, pandas/Polars | Standard | **Unverified** (standard practice) |

### Compute

A laptop with 16 GB RAM should handle 1.35M rows × \~50 features in LightGBM in minutes (**Unverified**; confirm in the roadmap spike). Free hosting: Streamlit Community Cloud gives roughly 0.08–2 CPU cores and 690 MB–2.7 GB RAM and sleeps after 12 hours idle ([Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) · **High**), so ship a small serialized model, not the training data. Hugging Face lists free CPU Basic as 2 vCPU / 16 GB, but its overview now says Gradio and Docker Spaces need a paid plan to create ([HF Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) · **High**).

## 6. Differentiation

Most public Lending Club projects stop at "XGBoost beats logistic regression, here is a SHAP beeswarm." Yours stands out if it reads like a bank's model development and validation pack: honest out-of-time numbers, a real points scorecard, Reg B-style reason codes that are checked for fidelity, and a fairness section that admits what the data cannot show. Gaps below are from sections 2 to 4; the "why it stands out" column is my judgement (**Medium**).

| Differentiator | What most projects do | What you would do | Evidence it matters |
| --- | --- | --- | --- |
| Leakage-proof, out-of-time result | Random split, post-origination fields, AUC 0.9+ | Application-time features only, train on early vintages, test on later ones, report AUC per vintage | [Qian et al.](https://arxiv.org/pdf/2112.10078); [Crumley](https://daniellecrumley.github.io/LendingClubProject/Data-Cleaning.html) |
| Champion vs challenger framing | One leaderboard | Logistic WoE scorecard = champion; LightGBM (monotone-constrained) = challenger; quantify the AUC/KS lift and the cost in interpretability | [Lessmann et al.](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) |
| Imbalance handled the calibrated way | SMOTE, then report accuracy | Show SMOTE vs no-resampling vs threshold moving, with Brier and calibration plots, and keep the calibrated one | [van den Goorbergh et al.](https://arxiv.org/abs/2202.09101v1) |
| Points scorecard a credit officer recognises | Rarely built | OptBinning monotonic bins, WoE, points-to-double-odds scaling, score bands with PD per band | [OptBinning](https://libraries.io/pypi/optbinning) |
| Reason codes to the Reg B standard | Global SHAP plots only | Top-4 adverse action reasons per declined applicant, from scorecard points-below-average and from SHAP, side by side, plus a fidelity test | [Reg B interpretation](https://www.consumerfinance.gov/rules-policy/regulations/1002/2026-07-21/interp-9/); [FinRegLab](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) |
| Explanation stability | Not checked | SHAP rank stability across seeds (Kendall's W), flag unstable mid-ranked features | [Lin & Wang 2025](https://arxiv.org/pdf/2508.01851) |
| Fairness with honest limits | Skipped or done with fake labels | Geography proxy on LC (clearly labelled), true sex/age audit on German Credit, Fairlearn mitigation, a mini LDA search | [Gillis et al.](https://facctconference.org/static/papers24/facct24-27.pdf); [Kozodoi et al.](https://arxiv.org/abs/2103.01907v3) |
| Business view | AUC only | Approval-rate vs bad-rate trade-off curve and expected-loss / profit at chosen cutoff | [German Credit cost matrix](https://archive.ics.uci.edu/dataset/144) |
| Validation pack | None | Model card + SR 26-2-style validation summary (conceptual soundness, outcomes analysis, monitoring incl. drift) | [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf); [Model Cards](https://arxiv.org/pdf/1810.03993) |
| Live decision app | Static notebook | App where a user enters an application and gets PD, score, decision, and four reasons | [Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) |

## 7. Evaluation design

Judge models on an out-of-time test set with a basket of ranking, calibration, business and fairness metrics, and report uncertainty. An honest outcome on Lending Club application-time data is an out-of-time AUC around 0.66 to 0.73, with gradient boosting a small step above a good logistic scorecard.

### Metrics

| Metric | Measures | Why a bank cares | Source · confidence |
| --- | --- | --- | --- |
| AUC (and Gini = 2·AUC − 1) | Ranking of defaulters above non-defaulters | Standard discrimination metric | [Lessmann et al.](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) · **High** |
| KS statistic | Largest gap between score distributions of goods and bads | Classic scorecard metric; also locates a cutoff | [SciPy ks\_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) · **High** |
| Brier score + calibration curve | Accuracy of the PD values themselves | PD drives pricing, capital and provisions | [Lessmann et al.](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) · **High** |
| H-measure (optional) | Discrimination under one shared cost distribution | Fixes AUC's inconsistent cost assumptions | [Lessmann et al.](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf) · **High** |
| Predicted vs realised default rate per score band | Back-test by grade | Basel requires this comparison at least annually | [BCBS CRE36](https://www.bis.org/committees/bcbs/basel-framework/81373/chapter.pdf) · **High** |
| Approval rate vs bad rate, expected profit/loss at cutoff | Business trade-off | How a credit policy is actually set | **Unverified** (industry practice) |
| Equalised-odds gap (TPR/FPR by group), approval-rate ratio | Group fairness | Separation is recommended for scorecards | [Kozodoi et al.](https://arxiv.org/abs/2103.01907v3) · **High**; [Fairlearn](https://fairlearn.org/main/api_reference/index.html) · **High** |
| SHAP rank stability (Kendall's W over seeds) and reason-code fidelity | Reliability of explanations | Reason codes must reflect the model | [Lin & Wang](https://arxiv.org/pdf/2508.01851) · **Medium**; [FinRegLab](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf) · **High** |

```latex
KS = \max_{s} \left| F_{\text{bad}}(s) - F_{\text{good}}(s) \right| = \max_{t} \left( \text{TPR}(t) - \text{FPR}(t) \right)
```

### Baselines

1. Naive: predict the training default rate for everyone (AUC 0.5; Brier sets the floor).
2. LendingClub's own `sub_grade` as a score (the incumbent to beat, run separately because it is a model output).
3. Logistic regression on WoE-binned features = the scorecard champion.
4. LightGBM, unconstrained and monotone-constrained = challengers.

### Validation method

- **Out-of-time split by `issue_d`:** for example train 2007–2015, validate 2016, test 2017–2018 vintages that have matured. Exact years are an open question for the roadmap spike (section 8).
- Hyperparameters tuned only on the validation window; the test window is touched once.
- **Uncertainty:** DeLong test or bootstrap for the AUC difference between champion and challenger ([pROC roc.test reference, DeLong et al. 1988](https://search.r-project.org/CRAN/refmans/pROC/html/roc.test.html) · **High**); bootstrap CIs for KS and Brier.
- **Stability:** AUC per vintage year and an adversarial-validation check of train vs test drift ([Qian et al.](https://arxiv.org/pdf/2112.10078) · **Medium**).

### What an honest result looks like

- Published application-time Lending Club AUCs: 0.6575 to 0.6644 with XGBoost on the Zenodo extract (cross-validated) ([Sanz-Guerrero & Arroyo](https://arxiv.org/html/2401.16458v3) · **High**), and 0.72 to 0.73 out-of-time on 2018–2020 ([Qian et al.](https://arxiv.org/pdf/2112.10078) · **Medium**). The difference likely reflects years and feature sets; treat 0.66–0.73 as the plausible range.
- Expect the challenger to beat the scorecard by a few AUC points, not tens (**Unverified**, consistent with [Lessmann et al.](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf)).
- Any AUC above about 0.80 on this data is a red flag for leakage until proven otherwise (**Unverified**, inference from the published range).
- Resampled models should show similar AUC and worse calibration than unresampled ones ([van den Goorbergh et al.](https://arxiv.org/abs/2202.09101v1) · **High**).

## 8. Pre-build research roadmap

About 10 hours, i.e. the first week, before writing the execution plan. Do items in order; each ends with a short note in a `research/` folder. Time estimates are mine (**Unverified**).

| # | Item (read, learn, test or verify) | Time | Deliverable | Exit criterion |
| --- | --- | --- | --- | --- |
| 1 | Download the [Zenodo LC extract](https://zenodo.org/records/11295916); profile columns, default rate, rows per `issue_d` year | 1.5 h | `data_profile.md` with a table of rows and default rate by year | You can say which vintages are fully matured and the default rate per year |
| 2 | Verify no leakage: compare the extract's columns with the post-origination list ([Crumley](https://daniellecrumley.github.io/LendingClubProject/Data-Cleaning.html)); decide on `grade`/`sub_grade`/`int_rate` | 1 h | `feature_whitelist.csv` with reason per column | Every feature is tagged "known at application" with a reason |
| 3 | Fix the out-of-time split and run a 20-minute logistic + LightGBM spike | 1.5 h | Split years + spike AUCs | Spike AUCs land inside 0.62–0.75; if above 0.80, return to item 2 |
| 4 | Read scorecard method: OptBinning tutorial, WoE/IV, points-to-double-odds scaling ([OptBinning](https://libraries.io/pypi/optbinning)) | 1.5 h | One-page scorecard spec (base score, PDO, bins rules) | You can hand-compute one applicant's score from a bin table |
| 5 | Read imbalance evidence ([van den Goorbergh](https://arxiv.org/abs/2202.09101v1), [Elor](https://arxiv.org/abs/2201.08528v3)) and LightGBM weighting notes | 0.5 h | Decision note: what imbalance arms to compare | Written list of 3 arms and the metrics that will decide |
| 6 | Read Reg B reason rules ([1002.9](https://www.law.cornell.edu/cfr/text/12/1002.9), [interpretation](https://www.consumerfinance.gov/rules-policy/regulations/1002/2026-07-21/interp-9/)) and FinRegLab summary | 1 h | Reason-code spec: method, max 4 reasons, wording map from features to plain-English reasons | Spec covers both scorecard and SHAP routes |
| 7 | SHAP scale and stability: [TreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), [Lin & Wang](https://arxiv.org/pdf/2508.01851) | 0.5 h | Note: explained scale, background size, stability test design | Choice of log-odds vs probability made and justified |
| 8 | Fairness design: [Kozodoi et al.](https://arxiv.org/abs/2103.01907v3), [Gillis et al.](https://facctconference.org/static/papers24/facct24-27.pdf), Fairlearn API; test joining LC 3-digit ZIP or state to a Census demographic table | 1.5 h | Fairness plan: groups, metrics, mitigation, proxy caveats | Join works for at least 95% of rows, or fall back to German Credit-only fairness (decision written) |
| 9 | Validation framing: skim [SR 26-2 attachment](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) and [Model Cards](https://arxiv.org/pdf/1810.03993) | 0.5 h | Model card + validation report outline | Outline has the three SR 26-2 pillars and the nine model-card sections |
| 10 | Deployment check: deploy a hello-world app with a 1 MB pickled model on Streamlit Community Cloud | 0.5 h | Live URL | App loads under the free memory limit |

## 9. Delivery requirements

Ship a public GitHub repo, a free Streamlit Community Cloud app, and a model card. Keep the deployed artefact small: the app loads a pre-trained model and a bin table, never the 167 MB training file.

### Deployment options

| Option | Free tier | Fit | Source · confidence |
| --- | --- | --- | --- |
| **Streamlit Community Cloud (recommended)** | \~0.08–2 cores, 690 MB–2.7 GB RAM, sleeps after 12 h idle; deploys from GitHub | Best for a scoring form + charts; easy | [Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) · **High** |
| Hugging Face Spaces | CPU Basic 2 vCPU / 16 GB, sleeps when idle; Gradio and Docker Spaces now need a paid plan to create, static Spaces free | Only if you already pay for PRO | [HF Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) · **High** |
| FastAPI on a container host (Render, Fly, Cloud Run) | Varies; not researched here | Shows API skills; more setup | **Unverified** |

### Repo structure (proposed)

```text
credit-risk-scorecard/
  README.md
  MODEL_CARD.md           # Mitchell et al. sections
  VALIDATION_REPORT.md    # SR 26-2 pillars
  pyproject.toml
  data/README.md          # how to download; no raw data committed
  research/               # Phase 1 notes from section 8
  src/credit_risk/
    data.py  features.py  split.py
    scorecard.py  gbm.py  calibrate.py
    metrics.py  explain.py  reasons.py  fairness.py
  notebooks/              # thin, call src/
  app/streamlit_app.py
  models/                 # small serialized artefacts
  tests/                  # leakage whitelist test, metric tests
  .github/workflows/ci.yml
```

### README contents

1. One-line result with numbers (out-of-time AUC, KS, Brier for champion and challenger) and the live app link.
2. Problem and who uses it (link section 1 sources).
3. Data: source, licence (CC BY 4.0), years, leakage policy.
4. Method: split, scorecard, GBM, calibration, imbalance arms.
5. Results table + calibration plot + approval/bad-rate curve.
6. Explainability: global SHAP, example reason codes, stability result.
7. Fairness: groups, metrics, mitigation, and what the data cannot show.
8. Limitations (selection bias, regime shift, proxy groups) and links to model card and validation report.
9. How to reproduce in one command.

### CV-bullet format

Action verb + what you built + method + quantified result + business meaning. Fill numbers only after the build. Templates:

- "Built a credit default model on 1.35M Lending Club loans; a monotone-constrained LightGBM raised out-of-time AUC from \[x\] to \[y\] (KS \[a\] to \[b\]) over a WoE logistic scorecard, cutting bad rate by \[z\] pp at a fixed \[n\]% approval rate."
- "Delivered Reg B-style adverse action reason codes from SHAP and scorecard points, validated for fidelity and stability across \[k\] seeds (Kendall's W = \[w\])."
- "Audited fairness with Fairlearn; in-processing cut the equalised-odds gap from \[p\] to \[q\] at \[r\] AUC cost; deployed a live scoring app on Streamlit."

## 10. Risks and open questions

The biggest risk is scope: 40 hours fits a champion/challenger model, scorecard, reasons, fairness audit and app only if every stretch idea (LLM text, TabPFN, survival, reject inference) stays out of the core plan.

### Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Scope creep past 40 hours | High | Unfinished app or README | Core vs stretch list fixed in Phase 2; stretch only after the app is live |
| Hidden leakage inflates results | Medium | Credibility loss in interviews | Whitelist test in CI; AUC > 0.80 triggers review (section 7) |
| Fairness proxy is weak or misleading | High | Over-claiming | Label as geographic proxy; put the rigorous audit on German Credit |
| Regulatory facts go stale (CFPB, SR 26-2, EU AI Act all changed in 2025–2026) | Medium | Outdated README claims | Cite dates; re-check before publishing |
| Free hosting limits (memory, sleep, HF paid gating) | Medium | App down during a recruiter visit | Small artefacts; screenshot + GIF in README as fallback |
| Dataset licence ambiguity for the full Kaggle LC mirror | Low | Repo takedown | Use the CC BY 4.0 Zenodo extract; never commit raw data |
| Strong explanations that are unstable | Medium | Reason codes change with the seed | Stability test; group correlated features into families ([FinRegLab](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf)) |

Likelihood and impact ratings are my judgement (**Unverified**).

### Open questions to settle in the roadmap

- [ ] Which vintages are mature enough for the test window, and how many loans does that leave?
- [ ] Include LC's `grade`/`sub_grade`/`int_rate` as features, or only as an incumbent benchmark?
- [ ] Does the Zenodo extract keep `zip_code` / `addr_state` for a geographic fairness proxy?
- [ ] Which reason-code method is primary: scorecard points-below-average or SHAP?
- [ ] Target audience emphasis: bank model risk/validation roles vs quant/DS roles (changes how much to invest in the validation report vs model tuning).
- [ ] Is a FastAPI endpoint worth adding alongside Streamlit for DS/ML-engineering roles?

## 11. Recommended skills, libraries and Claude tools

### Technical skills

- Leakage-safe feature pipelines and out-of-time validation (scikit-learn pipelines, time-based splits).
- WoE/IV binning and points scaling (OptBinning); logistic regression diagnostics.
- Gradient boosting tuning with monotone constraints (LightGBM); probability calibration (isotonic/Platt).
- SHAP on the right scale; fidelity and stability testing.
- Group fairness metrics and mitigation (Fairlearn reductions and post-processing).
- Statistical comparison of models (DeLong, bootstrap).
- Packaging, tests, GitHub Actions CI, Streamlit deployment.

### Domain skills

- PD/LGD/EAD vocabulary, default definitions ([BCBS CRE36](https://www.bis.org/committees/bcbs/basel-framework/81373/chapter.pdf)), and the IFRS 9 expected-loss idea ([EY](https://ey.com/en_gl/ifrs-technical-resources/impairment-of-financial-instruments-under-ifrs-9)).
- Scorecard practice: score bands, cutoffs, approval/bad-rate trade-offs, champion/challenger.
- Model risk management under [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf).
- US fair lending: adverse action ([12 CFR 1002.9](https://www.law.cornell.edu/cfr/text/12/1002.9)), disparate impact and LDAs ([Gillis et al.](https://facctconference.org/static/papers24/facct24-27.pdf)); EU AI Act high-risk status.

### Libraries

pandas or Polars, scikit-learn, LightGBM (XGBoost/CatBoost optional), OptBinning, SHAP, Fairlearn, SciPy, matplotlib/plotly, Streamlit, pytest. All named with sources in section 5.

### Claude skills, plugins and connectors available to you here

| Tool | Use in this project |
| --- | --- |
| Consensus connector (paper search) | Find and cite more peer-reviewed credit-scoring papers in Phase 2 |
| `data:explore-data` and `data:validate-data` skills | Profile the LC extract (roadmap item 1) and QA results before publishing |
| `data:statistical-analysis` skill | DeLong/bootstrap comparisons and stability tests |
| `dataviz` / `data:create-viz` skills | Calibration, KS and approval-vs-bad-rate charts in a consistent style |
| `academic-research-skills:academic-paper` (citation-check mode) | Verify every citation in README and model card |
| `code-review` and `security-review` skills | Review the repo before you make it public |
| GitHub connector | Create the repo and PRs from a thread; it failed to connect in this session, so reconnect it before Phase 3 |
| Claude Docs | Keep this report and the Phase 2 plan as living docs |

## 12. Handoff summary for Phase 2

Copy the block below into the Phase 2 prompt.

```text
PROJECT: Credit default prediction, bank-style (champion scorecard vs challenger GBM), portfolio grade.
PROFILE: intermediate Python/ML; 4 weeks x 10 h = 40 h; week 1 = research roadmap (section 8, ~10 h).

DATA
- Main: Lending Club granting-model extract, Zenodo DOI 10.5281/zenodo.11295916, CC BY 4.0,
  1,347,681 loans 2007-2018, application-time variables only, target default vs fully paid.
- Fairness side-data: UCI German Credit (1,000 rows, has sex/age, CC BY 4.0, cost matrix 5:1).
- Do not commit raw data.

MODELLING
- Out-of-time split by issue_d (train early vintages, validate next, test latest matured).
- Champion: WoE logistic scorecard (OptBinning), points scaling, score bands.
- Challenger: LightGBM with monotone constraints; calibrated.
- Imbalance: compare (a) no resampling + threshold, (b) class weights + recalibration, (c) SMOTE;
  choose by Brier/calibration, not accuracy.
- LC grade/sub_grade/int_rate: incumbent benchmark only (decide in roadmap).

EVALUATION
- AUC/Gini, KS, Brier + calibration curve, predicted vs realised default by band,
  approval-rate vs bad-rate curve and expected loss at cutoff.
- DeLong/bootstrap CIs; AUC per vintage; adversarial validation for drift.
- Honest range: out-of-time AUC ~0.66-0.73; >0.80 = leakage review.

EXPLAINABILITY
- Global SHAP (state log-odds vs probability), SHAP rank stability across seeds (Kendall's W).
- Reg B-style top-4 adverse action reasons per declined applicant from (a) scorecard points
  below average, (b) SHAP; fidelity test.

FAIRNESS
- LC: geographic proxy groups (labelled as proxy); German Credit: sex and age.
- Metrics: equalised-odds gap, approval-rate ratio (Fairlearn MetricFrame).
- Mitigation: ExponentiatedGradient or ThresholdOptimizer; small LDA-style search; report AUC cost.

DELIVERY
- GitHub repo (src/ package, tests incl. leakage whitelist, CI), README, MODEL_CARD.md,
  VALIDATION_REPORT.md structured on SR 26-2 (conceptual soundness, outcomes analysis, monitoring).
- Streamlit Community Cloud app: input form -> PD, score, decision, 4 reasons.
- 3 quantified CV bullets (templates in section 9).

OUT OF SCOPE (stretch only after app is live): LLM/BERT text features, TabPFN, survival/time-to-default,
reject inference, FastAPI.

REGULATORY CONTEXT (dated)
- SR 26-2 (17 Apr 2026) replaced SR 11-7.
- CFPB Circulars 2022-03 and 2023-03 withdrawn 12 May 2025; Reg B 1002.9 reasons still required.
- EU AI Act Annex III (credit scoring) applies from 2 Dec 2027 (Reg. (EU) 2026/1744).

OPEN QUESTIONS: test-window years; grade features; zip/state availability; primary reason-code
method; audience emphasis (model risk vs quant/DS); FastAPI yes/no.
```

## Sources

All pages below were opened during this research on 2026-11-06.

**Regulators and standards**

1. [BCBS Basel Framework CRE36, IRB minimum requirements](https://www.bis.org/committees/bcbs/basel-framework/81373/chapter.pdf)
2. [Federal Reserve SR 26-2 cover letter (17 Apr 2026)](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm)
3. [SR 26-2 attachment, Revised Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf)
4. [Federal Reserve SR 11-7 (2011, now superseded)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
5. [12 CFR 1002.9, Regulation B notifications (Cornell LII)](https://www.law.cornell.edu/cfr/text/12/1002.9)
6. [CFPB Regulation B official interpretation, section 9](https://www.consumerfinance.gov/rules-policy/regulations/1002/2026-07-21/interp-9/)
7. [CFPB withdrawn guidance list](https://www.consumerfinance.gov/compliance/guidance/withdrawn-guidance/)
8. [Federal Register, CFPB Circular 2023-03 (Apr 2024)](https://www.govinfo.gov/content/pkg/FR-2024-04-17/html/2024-08003.htm)
9. [CFPB proxy methodology (BISG) repo](https://www.github.com/cfpb/proxy-methodology)

**Legal and industry commentary**

10. [Consumer Finance Monitor on Circular 2022-03 (May 2022)](https://www.consumerfinancemonitor.com/2022/05/26/cfpb-issues-new-circular-on-the-application-of-ecoa-adverse-action-notice-requirements-to-credit-decisions-using-algorithms/)
11. [Krieg DeVault, CFPB withdraws 67 guidance documents](https://kriegdevault.com/insights/cfpb-withdraws-67-guidance-documents)
12. [Morgan Lewis CFPB guidance tracker](https://www.morganlewis.com/topics/cfpb-guidance-tracker-rescinded-and-remaining)
13. [Gibson Dunn, EU AI Act omnibus agreement](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/)
14. [Mondaq, EU Digital Omnibus on AI enters into force](https://www.mondaq.com/eu-digital-omnibus-on-ai-enters-into-force/1825722)
15. [EY, Impairment under IFRS 9](https://ey.com/en_gl/ifrs-technical-resources/impairment-of-financial-instruments-under-ifrs-9)
16. [myFICO, How lenders use credit scores](https://www.myfico.com/credit-education/credit-scores/how-lenders-use-credit-scores)
17. [Nasdaq / Motley Fool, LendingClub ends P2P platform (Oct 2020)](https://www.nasdaq.com/articles/lendingclub-is-ending-its-p2p-lending-platform-now-what-2020-10-08)
18. [FinRegLab, ML Explainability and Fairness: Insights from Consumer Lending (Jul 2023)](https://finreglab.org/wp-content/uploads/2023/07/FRL-ML-EWP_July2023.pdf)

**Papers**

19. [Lessmann, Seow, Baesens & Thomas, Benchmarking classification algorithms for credit scoring (EJOR 2015)](https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf)
20. [Kozodoi, Jacob & Lessmann, Fairness in Credit Scoring (arXiv 2103.01907)](https://arxiv.org/abs/2103.01907v3)
21. [Fuster et al., Predictably Unequal? (J. Finance 2022, SFI page)](https://www.sfi.ch/de/publications/predictably-unequal-the-effects-of-machine-learning-on-credit-markets.)
22. [Gillis, Meursault & Ustun, Operationalizing the Search for LDAs (FAccT 2024)](https://facctconference.org/static/papers24/facct24-27.pdf)
23. [Namvar et al., Credit risk prediction in an imbalanced social lending environment](https://arxiv.org/pdf/1805.00801)
24. [Qian et al., Managing dataset shift by adversarial validation for credit scoring](https://arxiv.org/pdf/2112.10078)
25. [Sanz-Guerrero & Arroyo, Credit Risk Meets LLMs (arXiv 2401.16458)](https://arxiv.org/html/2401.16458v3)
26. [Feng et al., Empowering Many, Biasing a Few (CALM)](https://arxiv.org/abs/2310.00566v3)
27. [TabPFN: Accurate predictions on small data with a tabular foundation model (Nature 2025)](https://pubmed.ncbi.nlm.nih.gov/39780007/)
28. [TabArena: A Living Benchmark for ML on Tabular Data (NeurIPS 2025)](https://arxiv.org/pdf/2506.16791)
29. [Grinsztajn et al., Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
30. [van den Goorbergh et al., The harm of class imbalance corrections](https://arxiv.org/abs/2202.09101v1)
31. [Elor & Averbuch-Elor, To SMOTE, or not to SMOTE?](https://arxiv.org/abs/2201.08528v3)
32. [Lin & Wang, SHAP Stability in Credit Risk Management (2025)](https://arxiv.org/pdf/2508.01851)
33. [Ballegeer, Bogaert & Benoit, Stability of explanations in IDCS credit scoring (EJOR 2025)](https://ideas.repec.org/a/eee/ejores/v326y2025i3p630-640.html)
34. [Botha et al., Term-structure of write-off risk under IFRS 9 (arXiv 2026)](https://arxiv.org/pdf/2603.11897)
35. [Mitchell et al., Model Cards for Model Reporting (FAT\* 2019)](https://arxiv.org/pdf/1810.03993)

**Datasets**

36. [Zenodo, Lending Club loan dataset for granting models](https://zenodo.org/records/11295916)
37. [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144)
38. [Freddie Mac Single-Family Loan-Level Dataset](https://freddiemac.com/research/datasets/sf-loanlevel-dataset)
39. [Credit-score book, Appendix C dataset catalog](https://mikenguyen13.github.io/credit_score/appendices/C-datasets.html)

**Projects and repos**

40. [deepsense.ai, Home Credit competition write-up](https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/)
41. [kozodoi/Kaggle\_Home\_Credit](https://github.com/kozodoi/Kaggle_Home_Credit)
42. [gegznav Lending Club project](https://gegznav.github.io/ds-projects/lending-club)
43. [andrewhnberry/Identifying-bad-debt](https://github.com/andrewhnberry/Identifying-bad-debt)
44. [Crumley, Lending Club data cleaning](https://daniellecrumley.github.io/LendingClubProject/Data-Cleaning.html)
45. [Survival analysis on Lending Club loans (2016 blog)](https://systemsforecasting.com/2016/07/application-of-survival-analysis-to-p2p-lending-club-loans-data/)

**Library and platform docs**

46. [OptBinning on libraries.io](https://libraries.io/pypi/optbinning)
47. [LightGBM parameters (v4.4.0)](https://lightgbm.readthedocs.io/en/v4.4.0/Parameters.html)
48. [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
49. [Fairlearn API reference](https://fairlearn.org/main/api_reference/index.html)
50. [SciPy ks\_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
51. [pROC roc.test (DeLong reference)](https://search.r-project.org/CRAN/refmans/pROC/html/roc.test.html)
52. [Streamlit Community Cloud, manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
53. [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)

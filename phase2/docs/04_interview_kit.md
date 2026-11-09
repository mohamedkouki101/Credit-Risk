# Interview kit (PLAN §12 in full)

Rule: **every number in brackets comes from `results/*.csv` after the build. Never fill one in from memory or from this plan.**

## 12.1 CV bullet templates

1. **Model and result.** "Built a bank-style credit default model on [N] Lending Club loans with a strict out-of-time test (train [years], test [year]). A monotone-constrained LightGBM challenger [raised / did not raise] AUC from [x] to [y] (95% CI on Δ: [a, b]) over a WOE logistic scorecard. At a fixed [p]% approval rate, the bad rate [fell / changed] by [z] pp."
   - *Null-result version:* "…the challenger's AUC gain of [Δ] was not significant (CI [a, b]), so the interpretable scorecard was retained as champion, saving [k] reason-code families of complexity."
2. **Explainability and compliance.** "Implemented Regulation B-style adverse-action reason codes from scorecard points and SHAP. In a fidelity test, the top reason moved PD more than a random feature for [q]% of declined applicants. Top-10 ranking stability across [10] seeds: Kendall's W = [w]."
3. **Fairness, model risk and deployment.** "Audited fairness on a [proxy / age] grouping with Fairlearn: approval-rate ratio [r], equalised-odds gap [g]. Mitigation cut the gap to [g'] for [ΔAUC] AUC. Shipped a FastAPI scoring service and a Streamlit app with an SR 26-2-structured validation report and PSI drift monitoring."

## 12.2 Twenty-five probing questions with answer outlines

**Data and target**
1. **How exactly did you define default, and why not 90 DPD?** Charge-off over the full 36-month term (statuses listed). The file has no delinquency history, so 90 DPD is unobservable. Consequence: PDs are lower than Basel PDs. Basel's definition is in CRE36.
2. **How did you avoid right-censoring?** Only loans whose contractual maturity is before the snapshot (36-month loans issued ≤ Dec 2015). Indeterminate loans are counted and given a sensitivity analysis. Recent vintages are used only for unlabelled drift monitoring.
3. **Which features did you exclude for leakage, and how do you know you caught them all?** Whitelist, not blacklist. Regex test in CI. Missingness-by-target audit. Adversarial validation. The AUC > 0.80 tripwire. Mention `last_fico_range` as the classic trap.
4. **Why not use LendingClub's grade or interest rate?** They are outputs of LC's own model, so using them means partly re-learning their score. They serve as the incumbent benchmark. Mention `installment` encodes `int_rate`.
5. **What selection bias does this data have, and what would you do in a bank?** No rejects. Reject inference (augmentation, parcelling, self-learning). Evaluating on accepts misleads, so use the kickout measure. In a bank: a randomised low-cutoff "approve a few below cutoff" test sample.

**Validation and statistics**
6. **Why out-of-time instead of k-fold?** Production scores future applicants. Drift exists (show the adversarial AUC). Random folds leak future regimes into training.
7. **Is the challenger's gain real?** DeLong plus a paired bootstrap CI on ΔAUC. The pre-registered minimum effect. Also ΔBrier and business ΔEL. Significance is not materiality.
8. **Explain KS and how it relates to AUC.** KS = max(TPR − FPR) = max gap between the score CDFs of goods and bads. Both are rank-based. KS is a single point, AUC integrates over all points. KS shows where the separation is greatest.
9. **Why report PR-AUC if AUC is the standard?** PR-AUC depends on the base rate and focuses on the positive class. Useful when the bad rate shifts between vintages. Explain why it shouldn't be compared across years with different bad rates.
10. **Your model is well ranked but miscalibrated. Does it matter?** Yes for pricing, expected loss, IFRS 9 provisions and capital. Show the calibration slope/intercept. Explain Platt vs isotonic and why sigmoid preserves the ranking.
11. **What would a null result have looked like, and what would you have done?** Recite the pre-registration: champion retained, documented. That is a valid model-risk outcome.

**Imbalance**
12. **Why didn't you use SMOTE?** It was tested (arm D). It distorts the PDs ([van den Goorbergh](https://arxiv.org/abs/2202.09101v1)). Give your own validation table. The decision rule picks by log-loss.
13. **Is a ~[bad rate]% default rate even "imbalanced"?** Moderately. The real issue is the cost asymmetry, which belongs in the cutoff, not in the training distribution.

**Scorecard**
14. **Walk me through WOE and IV.** WOE = ln(%good/%bad) per bin (state the sign convention used). IV = Σ(%good − %bad)·WOE. Monotone binning gives business sense and stability. IV thresholds are heuristics.
15. **Derive points-to-double-odds scaling.** Factor = PDO/ln 2; offset = base − factor·ln(base odds); score = offset + factor·ln(odds). Do the 600 / 50:1 / 20 example live on a whiteboard.
16. **Why monotone constraints on the GBM?** Business sense, regulator expectations, stability, reason codes that make sense. Cost measured in AUC (ablation A-02). LightGBM's `basic` method over-constrains, while `intermediate` and `advanced` constrain less.

**Explainability**
17. **What exactly do your SHAP values explain?** Log-odds (raw margin) with an interventional background of 500 rows. Why not probability: additivity holds in log-odds, and probability needs interventional mode. Correlated features distort attributions, which is why reasons are grouped into families.
18. **How do you know your reason codes are right?** The fidelity test, the stability test, and agreement with the scorecard method. Reg B says reasons must "accurately describe the factors actually considered".
19. **Scorecard reasons and SHAP reasons disagree for an applicant. Which do you send?** The reasons of the model that made the decision. Disagreement is itself a model-risk finding, so quantify the agreement rate.

**Fairness and regulation**
20. **You have no race field. How can you say anything about fairness?** A geographic proxy (area composition, ecological fallacy) and age on South German Credit. Be explicit about what cannot be concluded. Explain why BISG wasn't possible.
21. **Disparate impact was removed from Reg B in 2026. Why measure it?** The effects test was removed effective 21 Jul 2026, and the rule is under challenge. Disparate treatment via proxies, state laws, FHA and CRA remain. Lenders still monitor fair lending, and good model risk practice requires knowing your model's group behaviour. Date these claims.
22. **Your mitigation used group-specific thresholds. Is that legal?** It uses the protected attribute at decision time, which raises disparate-treatment risk. That's why it is shown only as a benchmark, and in-processing / LDA search is the deployable option.

**Model risk, deployment and process**
23. **How would a validator challenge this model under SR 26-2?** Conceptual soundness (data, target, binning choices, benchmarks), outcomes analysis (OOT, back-testing by band), ongoing monitoring (PSI with critical values). Limitations. Materiality tiering.
24. **What happens when PSI goes red in production?** Check which features drifted. Check whether performance degraded (that needs outcomes, which lag). Recalibrate vs rebuild. Reg B requires periodic revalidation. Rule-of-thumb PSI thresholds aren't statistical, so use critical values.
25. **How much of this did AI write, and how do you know it's correct?** Be honest about what Claude Code did. Point to the tests with known answers (synthetic data, PDO property test, SHAP additivity), the hand-computed scorecard example, and the pre-registration tag. (Upstart's posting explicitly values agentic tooling, [J11](https://careers.upstart.com/jobs/data-science-intern).)

## 12.3 The five weaknesses to disclose proactively

1. **Charge-off target, not Basel default.** PDs are not comparable to 90-DPD PDs.
2. **Selection bias.** Approved loans only. Reject inference is discussed, and the experiment is a stretch task.
3. **Historical, ended platform.** 2007–2015 performance data. LC retail notes ended in 2020. No claim about today's borrowers.
4. **Fairness groups are proxies.** Area-level composition for LC, a 1,000-row dataset for age. Small n means wide CIs.
5. **Single-snapshot outcome.** Only the 2018Q4 snapshot status is observed, so timing-based analyses (24-month target, survival) rest on payment-date approximations.

## 12.4 Three-minute talk track

1. *(20 s) Problem:* "Lenders must rank applicants by default risk and, by law, tell declined applicants why. I built the kind of champion/challenger package a bank's model risk team would review."
2. *(30 s) Data and honesty:* "[N] Lending Club loans. Application-time features only via a whitelist enforced in CI. Default = charge-off over the full term, using only matured vintages. Out-of-time test on [year], touched once. I pre-registered the success criteria in git before training."
3. *(40 s) Models:* "Champion: a WOE logistic scorecard with points-to-double-odds scaling, the kind a credit officer reads. Challenger: monotone-constrained LightGBM, calibrated. Result: [AUC/KS for both, CI on Δ]. At [p]% approval, [bad rate change, EL change]."
4. *(30 s) Explanations:* "Reason codes the Reg B way, from scorecard points and from SHAP. Then I tested them: the top reason beats a random feature [q]% of the time, and they are stable across seeds (W = [w])."
5. *(30 s) Fairness and risk:* "Fairness on a labelled proxy and on age: gaps of [g], mitigation cost [ΔAUC]. Legal context dated to 2026. Validation report on the SR 26-2 pillars and PSI monitoring with proper critical values."
6. *(30 s) Demo and limits:* show the app (score + 4 reasons). "Main limits: no rejects, charge-off target, historical platform. Next I'd run the reject-inference experiment."

## 12.5 Ten-minute talk track (slide by slide, ≈1 min each)

1. Problem, user and the decision being supported (who acts on the score).
2. Data: source, licence, target definition diagram (issue → maturity → snapshot), exclusions with counts.
3. Leakage defences: whitelist, tests, adversarial validation, AUC tripwire. *Show the CI test.*
4. Split design: development/validation/OOT/monitoring timeline, rolling-origin backtest table.
5. Champion scorecard: one bin table, the PDO formula, a hand-scored example applicant.
6. Challenger + imbalance: arms table, decision rule, calibration curve before and after.
7. Results: OOT table with CIs, approval vs bad-rate curve, EL at cutoff. The pre-registered decision applied.
8. Explainability: global SHAP vs IV, reason-code agreement, fidelity and stability results.
9. Fairness: metric table, trade-off plot, legal context (2026 changes), what cannot be concluded.
10. Model risk and deployment: SR 26-2 mapping, PSI on 2016–2018, architecture diagram, live demo, the 5 weaknesses, next steps.

# Target audience research: what screeners look for

Assumed target roles: quant research, data science, and risk / model-validation / ML engineering internships at banks, asset managers, quant funds and fintech lenders. Quant trading is included, but this project is a weak direct signal for it (see "Honest fit" below).

All pages below were opened on 2026-11-06. Several postings had already closed. Their text still shows what these teams ask for.

## Sources opened

| ID | Source | What it says that matters here | Confidence |
|---|---|---|---|
| J1 | [JPMorgan 2027 Quantitative Research, Risk & Treasury summer internship (via Nabla Careers)](https://www.nablacareers.com/finance-jobs/jp-morgan/north-america/risk-management/2027-quantitative-research-risk-and-treasury-summer-internship-associate-united-states/) | "back testing and assess strategies"; "Assess models for soundness, risks, and enhancements". Model Risk Governance team: "validate models, draft reports, interact with developers". Wholesale Credit team: "develop credit risk models, apply AI/ML". Requires Python and/or C++, "Excellent communication skills for complex concepts". Preferred: SQL, "Familiarity with AI tools in research and programming". | Medium (job-board copy of the posting) |
| J2 | [Two Sigma Quantitative Researcher Intern, 2027 Summer](https://careers.twosigma.com/careers/JobDetail/New-York-New-York-United-States-Quantitative-Researcher-Intern-2027-Summer/13945) | Requires an "in-depth research project, examining real-world data"; "independent thinker" who can "communicate complex ideas clearly"; "You don't need a background in finance". | High (official) |
| J3 | [Two Sigma Data Scientist Internship (Summer 2026)](https://careers.twosigma.com/careers/JobDetail/New-York-New-York-United-States-Data-Scientist-Internship-Summer-2026/13585) | "Proficient in Python and SQL"; "Conduct literature reviews to develop and apply cutting-edge methodologies"; "rigorously explore and test your theories". | High (official) |
| J4 | [Two Sigma careers: Quantitative Research & Data Science](https://www.twosigma.com/careers/quantitative-research-data-science/) | "Systematically test and expand hypotheses"; "scientific thinking, curiosity, and precision". | High (official) |
| J5 | [Citadel: Our Quantitative Research Interview Process](https://www.citadel.com/careers/career-perspectives/our-quantitative-research-interview-process/) | Interviewers ask about school projects and internships and "how your past internship has contributed to business outcomes". "Share your thought process"; "Discuss trade-offs among different possible approaches". | High (official) |
| J6 | [Citadel Securities 2026–27 Quantitative Research Analyst internship (GradConnection)](https://au.gradconnection.com/employers/citadel-securities/jobs/citadel-securities-2026-2027-quantitative-research-analyst-internship) | "translate algorithms into code"; backtest and deploy models; probability, statistics and ML; "Independent research experience"; "close attention to detail"; strong written and verbal communication. | Medium (job board) |
| J7 | [Jane Street: What to expect (interviewing)](https://www.janestreet.com/join-jane-street/interviewing/) | "asking great questions is more important than knowing all the answers". ML interviews involve "realistic modeling problems". QT interviews won't test finance knowledge. | High (official) |
| J8 | [Jane Street QR 2027 summer internship (Bond University job board)](https://joblink.bond.edu.au/campaign/jane-street-quantitative-research-2027-summer-internship) | "eager to ask questions, admit mistakes, and learn new things"; "strong programmer who's comfortable with Python"; "from initial exploration to finding and productionizing a signal". | Medium (job board) |
| J9 | [Goldman Sachs Finance & Risk Quantitative Strats summer analyst (job board copy)](https://www.jobsearchplace.co.uk/jobs/2026-emea-london-finance-and-risk-quantitative-strats-summer-analyst-3034736803) | "construct quantitative models"; "research, analysis and aptitude for innovation". No specific tools listed. | Low (aggregator) |
| J10 | [Capital One Master's Data Science Internship, Summer 2027 (The Muse)](https://www.themuse.com/jobs/capitalone/current-masters-data-science-internship-summer-2027) | 6+ months of Python/R and of ML. Preferred: inferential statistics, SQL, large-scale data. "share technical ideas that resonate with audience needs"; willingness to test tools "even when they fail". | Medium (job board) |
| J11 | [Upstart Data Science Intern](https://careers.upstart.com/jobs/data-science-intern) | Lending ML: "underwriting and pricing", fair lending and explainability. "Understand the underpinnings of ML models and how they create value for the business". "Communicate findings clearly to get buy-in". "Ability to use modern agentic tooling (Claude Code, Codex, …) efficiently". | High (official) |
| J12 | [Nissan Credit Risk Analytics Intern, Summer 2027](https://alliance.wd3.myworkdayjobs.com/en-GB/nissanjobs/job/Irving-Texas---United-States-of-America/Credit-Risk-Analytics-Intern---Summer-2027---Irving--TX_R00214224) | "Analyze portfolio performance and credit risk trends"; dashboards on risk metrics; "Support predictive modeling and forecasting initiatives related to credit performance"; policy-change analysis. | High (official) |
| J13 | [Goldman Sachs AM Quantitative Investing summer analyst (Canary Wharfian)](https://www.canarywharfian.co.uk/jobs/goldman-sachs/2027-americas-new-york-city-area-asset-management-quantitative-investing-summer-analyst/5b431b98-b428-42d1-bd05-671d58f2c734) | Lists no skills. Recorded only to show that asset-manager signals are **not** well sourced here. | Low |

## Signals the project must send

Each signal says where it comes from and which plan elements deliver it. Task IDs refer to docs/03_roadmap.md.

| # | Signal | Evidence | Plan elements that deliver it |
|---|---|---|---|
| S1 | **An in-depth research project on real-world data**, run on 1M+ loans, not a toy | J2, J3, J6 | Full LC file (D-01…D-06); out-of-time design; experiment log (§7); results report (DOC-02) |
| S2 | **Scientific method: hypotheses tested and pre-registered, including null results** | J3, J4 | PLAN §2 pre-registered criteria, committed before modelling (task P-01, git tag `prereg-v1`); null-result templates |
| S3 | **Backtesting and out-of-time thinking** | J1, J6 | OOT split + rolling-origin backtest across vintages (M-07); per-vintage AUC/KS; PSI monitoring (MR-02) |
| S4 | **Assessing models for soundness, plus validation reports** | J1 (Model Risk Governance) | VALIDATION_REPORT.md on the three SR 26-2 pillars (DOC-03); champion vs challenger; model card (DOC-01) |
| S5 | **Credit risk modelling with AI/ML, done the way lenders do it** | J1 (Wholesale Credit), J11, J12 | WOE/IV scorecard with PDO scaling (M-03…M-05); monotone LightGBM (M-06); expected loss and approval/bad-rate curves (EV-03) |
| S6 | **Statistics depth: inference, uncertainty, calibration** | J6, J10, J11 | DeLong and paired bootstrap CIs (EV-02); calibration analysis (M-08); PSI critical values (MR-02); the synthetic known-answer test (T-05) |
| S7 | **Clean, tested, production-quality Python, with SQL** | J1, J3, J6, J8, J10 | `src/` package, pytest suite with leakage and metric tests (§9), CI, type hints. DuckDB SQL for vintage tables (D-04) |
| S8 | **Translating models into code and deploying them** | J6, J8 | FastAPI scoring service + Streamlit app (DEP-01…DEP-05); pinned model artefact with hash |
| S9 | **Explainability and fair lending, the credit-specific kind** | J11 | Reg B reason codes from two methods, fidelity and stability tests (X-01…X-05); fairness audit with legal context (F-01…F-06) |
| S10 | **Business impact framing** | J5, J11, J12 | Approval rate vs bad rate at cutoff; expected loss; "what the bank gains" line in README and talk tracks |
| S11 | **Trade-off reasoning you can explain out loud** | J5, J7 | "Why X not Y" table (§3); imbalance comparison (M-09); ablations (§7); interview kit Q&A |
| S12 | **Intellectual honesty: admitting mistakes and limits** | J7, J8 | Honesty protocol (§8); 5 weaknesses disclosed up front (§12); limitations section; null-result presentation |
| S13 | **Communicating complex ideas to non-specialists** | J1, J2, J10, J11 | Plain-language explanation in the app; 3-min and 10-min talk tracks; optional blog post |
| S14 | **Literature-grounded methods** | J3 | Every method choice cites a paper (SOURCES.md); a reproduction of a published Zenodo AUC (M-02) |
| S15 | **Fluent, honest use of AI tools** | J1, J11 | README "How this was built" note: what Claude Code did and what you checked by hand. Interview answer prepared (Q25) |

## Honest fit by role

- **Bank risk / model validation / credit DS (JPM QR Risk, Capital One, Upstart, Nissan):** strong direct fit. Weight the validation report, the scorecard and the reason codes heavily.
- **Quant research and data science at funds (Two Sigma, Citadel):** they want research process (S1, S2, S6, S14) more than domain knowledge. They state that finance knowledge is not required. This project shows that process. Expect to be probed on statistics, not on credit.
- **Quant trading (Jane Street, Citadel Securities):** weak direct signal. These interviews test problem solving and probability, not projects (J7). The project helps mainly as evidence of rigour. Do not over-invest in it for QT applications.
- **Asset managers:** no posting I opened listed concrete skills (J13). Treat the signals as Medium confidence for this group.

## What screeners penalise (implied by the same sources)

These are my inferences from the signals above (**Unverified** as stated claims):

- Inflated metrics with no out-of-time test. This contradicts S2, S3 and S12.
- Notebooks with no tests or packaging. This contradicts S7 and S8.
- Results with no business meaning. This contradicts S10.
- Being unable to explain why a choice was made. This contradicts S11 and J5's "trade-offs".

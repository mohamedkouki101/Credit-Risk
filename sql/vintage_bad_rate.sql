-- Vintage table: core population (data plan 5.3a) by issue quarter.
-- Input table `loans`: the cleaned, targeted frame (columns issue_d, y, segment,
-- exclusion_reason). Run with credit_risk.data.sql.run_sql("vintage_bad_rate", loans=df).
-- Must agree with credit_risk.data.target.vintage_profile (tests/unit/test_sql.py).
-- bad_rate = bad / (bad + good); indeterminate loans are counted but not in the rate.
SELECT
    CAST(year(issue_d) AS VARCHAR) || 'Q' || CAST(quarter(issue_d) AS VARCHAR) AS quarter,
    COUNT(*) AS rows,
    SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END) AS n_bad,
    SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END) AS n_good,
    SUM(CASE WHEN exclusion_reason = 'indeterminate' THEN 1 ELSE 0 END) AS n_indeterminate,
    SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN y IN (0, 1) THEN 1 ELSE 0 END), 0) AS bad_rate
FROM loans
WHERE segment = 'core'
GROUP BY 1
ORDER BY 1;

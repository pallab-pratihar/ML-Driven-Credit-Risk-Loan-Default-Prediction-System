-- ============================================================================
-- Credit Risk Portfolio Analytics
-- Engine: DuckDB (also runs on SQLite with minor syntax changes)
-- Table:  loans  -- registered from data/processed/credit_risk_featured.csv
-- Each query is delimited by a "-- name: <query_name>" header; the runner in
-- src/analytics/runner.py splits the file on those headers.
-- ============================================================================

-- name: 01_portfolio_overview
-- Headline portfolio KPIs. One loan per customer in this dataset, so the
-- customer count and the loan count are equal; both are reported explicitly.
SELECT
    COUNT(*)                                  AS total_customers,
    COUNT(*)                                  AS total_loans,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                  AS avg_loan_amount,
    ROUND(AVG(person_income), 2)              AS avg_income,
    ROUND(SUM(loan_amnt), 2)                  AS total_exposure,
    ROUND(SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END), 2) AS defaulted_exposure
FROM loans;

-- name: 02_default_rate_by_employment
-- Employment length is the stability proxy available in this dataset.
SELECT
    CASE
        WHEN person_emp_length IS NULL THEN 'Unknown'
        WHEN person_emp_length < 1  THEN '<1 year'
        WHEN person_emp_length < 3  THEN '1-3 years'
        WHEN person_emp_length < 6  THEN '3-6 years'
        WHEN person_emp_length < 11 THEN '6-10 years'
        ELSE '10+ years'
    END                                       AS employment_band,
    COUNT(*)                                  AS customers,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                  AS avg_loan_amount
FROM loans
GROUP BY 1
ORDER BY default_rate_pct DESC;

-- name: 03_default_rate_by_income_group
SELECT
    income_group,
    COUNT(*)                                  AS customers,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                  AS avg_loan_amount,
    ROUND(AVG(loan_percent_income) * 100, 2)  AS avg_loan_to_income_pct
FROM loans
GROUP BY 1
ORDER BY default_rate_pct DESC;

-- name: 04_default_rate_by_loan_intent
-- loan_intent is this dataset's "loan type" / product purpose dimension.
SELECT
    loan_intent,
    COUNT(*)                                  AS loans,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                  AS avg_loan_amount,
    ROUND(SUM(loan_amnt), 2)                  AS total_exposure
FROM loans
GROUP BY 1
ORDER BY default_rate_pct DESC;

-- name: 05_default_rate_by_credit_grade
-- loan_grade is the lender-assigned credit grade (A best .. G worst).
SELECT
    loan_grade,
    COUNT(*)                                  AS loans,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_int_rate), 2)              AS avg_interest_rate,
    ROUND(AVG(loan_amnt), 2)                  AS avg_loan_amount
FROM loans
GROUP BY 1
ORDER BY loan_grade;

-- name: 06_default_rate_by_home_ownership
SELECT
    person_home_ownership,
    COUNT(*)                                  AS customers,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(person_income), 2)              AS avg_income
FROM loans
GROUP BY 1
ORDER BY default_rate_pct DESC;

-- name: 07_default_rate_by_age_group
SELECT
    age_group,
    COUNT(*)                                  AS customers,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(cb_person_cred_hist_length), 2) AS avg_credit_history_years
FROM loans
WHERE age_group IS NOT NULL
GROUP BY 1
ORDER BY age_group;

-- name: 08_default_rate_by_prior_default
-- Bureau flag: has the customer defaulted before?
SELECT
    CASE WHEN has_prior_default = 1 THEN 'Prior default on file' ELSE 'Clean bureau file' END AS bureau_status,
    COUNT(*)                                  AS customers,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_int_rate), 2)              AS avg_interest_rate
FROM loans
GROUP BY 1
ORDER BY default_rate_pct DESC;

-- name: 09_loan_burden_bands
-- Loan-to-income burden is the strongest affordability lever a lender controls.
SELECT
    CASE
        WHEN loan_percent_income < 0.10 THEN '<10%'
        WHEN loan_percent_income < 0.20 THEN '10-20%'
        WHEN loan_percent_income < 0.30 THEN '20-30%'
        WHEN loan_percent_income < 0.40 THEN '30-40%'
        ELSE '40%+'
    END                                       AS loan_to_income_band,
    COUNT(*)                                  AS loans,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                  AS avg_loan_amount
FROM loans
GROUP BY 1
ORDER BY default_rate_pct;

-- name: 10_high_risk_customers
-- Rule-based watchlist (independent of the ML model): heavy loan burden plus
-- either a weak credit grade or a prior default on the bureau file.
SELECT
    COUNT(*)                                  AS high_risk_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loans), 2) AS pct_of_portfolio,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(SUM(loan_amnt), 2)                  AS exposure_at_risk
FROM loans
WHERE loan_percent_income >= 0.30
  AND (loan_grade IN ('D', 'E', 'F', 'G') OR has_prior_default = 1);

-- name: 11_portfolio_distribution_by_loan_size
SELECT
    loan_amount_group,
    COUNT(*)                                  AS loans,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loans), 2) AS pct_of_loans,
    ROUND(SUM(loan_amnt), 2)                  AS total_exposure,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct
FROM loans
WHERE loan_amount_group IS NOT NULL
GROUP BY 1
ORDER BY total_exposure DESC;

-- name: 12_worst_segments_grade_x_intent
-- Cross-segment view used to prioritise policy changes. Segments with fewer
-- than 100 loans are excluded so the rates are not noise.
SELECT
    loan_grade,
    loan_intent,
    COUNT(*)                                  AS loans,
    ROUND(AVG(loan_status) * 100, 2)          AS default_rate_pct,
    ROUND(SUM(loan_amnt), 2)                  AS total_exposure
FROM loans
GROUP BY 1, 2
HAVING COUNT(*) >= 100
ORDER BY default_rate_pct DESC
LIMIT 10;

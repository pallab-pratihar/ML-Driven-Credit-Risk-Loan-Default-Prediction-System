"""Credit-risk feature engineering.

All transformations here are **row-wise**: every engineered column is a
function of that customer's own attributes only. No column statistic (mean,
quantile, target encoding, ...) is used, so these features can safely be
computed before the train/validation/test split without leaking information.

No feature uses the target, and none uses post-origination information such as
repayment behaviour -- the dataset only contains application-time attributes
plus the bureau flag, which is known before the credit decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Feature -> one-line rationale. Rendered into the notebooks and the README.
FEATURE_DOCS: dict[str, str] = {
    "loan_to_income_ratio": "Loan amount / annual income. Core affordability measure: how many years of gross income the loan represents.",
    "annual_interest_cost": "loan_amnt * interest rate. Absolute yearly cost of carrying the loan.",
    "interest_to_income": "Annual interest cost / income. Interest burden independent of principal repayment.",
    "est_monthly_payment": "Amortised monthly instalment assuming a 60-month term (term is not in the dataset; the assumption is constant across customers so it only rescales the feature).",
    "payment_to_income": "Estimated annual instalments / income. Debt-service proxy -- the closest available stand-in for a true DTI, since no other-debt column exists.",
    "credit_history_ratio": "Credit-history length / years since age 18. Captures how early the customer entered the credit system.",
    "employment_stability": "Employment length / years since age 18. Share of adult life spent in recorded employment.",
    "income_per_credit_year": "Income / (1 + credit history length). Distinguishes high earners who are new to credit from long-established borrowers.",
    "log_income": "log1p(income). Tames the strong right skew for the linear model.",
    "has_prior_default": "1 if the credit bureau file records a previous default, else 0.",
    "age_group": "Business-defined age band (fixed edges, not data-derived).",
    "income_group": "Business-defined income band (fixed edges, not data-derived).",
    "loan_amount_group": "Business-defined loan-size band (fixed edges, not data-derived).",
}

ASSUMED_TERM_MONTHS: int = 60


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of inf on zero/negative."""
    denom = denominator.where(denominator > 0)
    return numerator / denom


def _amortised_monthly_payment(
    principal: pd.Series, annual_rate_pct: pd.Series, months: int = ASSUMED_TERM_MONTHS
) -> pd.Series:
    """Standard annuity instalment; falls back to straight-line at 0% rate."""
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (1 + monthly_rate) ** months
        payment = principal * monthly_rate * factor / (factor - 1)
    straight_line = principal / months
    return payment.where(monthly_rate > 0, straight_line)


def add_credit_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered credit-risk features added.

    Columns that the source dataset does not provide are skipped silently, so
    the function degrades gracefully on a different loan dataset.
    """
    out = df.copy()

    has = out.columns.__contains__

    if has("loan_amnt") and has("person_income"):
        out["loan_to_income_ratio"] = _safe_divide(out["loan_amnt"], out["person_income"])

    if has("loan_amnt") and has("loan_int_rate"):
        out["annual_interest_cost"] = out["loan_amnt"] * out["loan_int_rate"] / 100.0
        out["est_monthly_payment"] = _amortised_monthly_payment(
            out["loan_amnt"], out["loan_int_rate"]
        )
        if has("person_income"):
            out["interest_to_income"] = _safe_divide(
                out["annual_interest_cost"], out["person_income"]
            )
            out["payment_to_income"] = _safe_divide(
                out["est_monthly_payment"] * 12.0, out["person_income"]
            )

    if has("cb_person_cred_hist_length") and has("person_age"):
        adult_years = (out["person_age"] - 18).clip(lower=1)
        out["credit_history_ratio"] = (
            out["cb_person_cred_hist_length"] / adult_years
        ).clip(upper=1.5)

    if has("person_emp_length") and has("person_age"):
        adult_years = (out["person_age"] - 18).clip(lower=1)
        out["employment_stability"] = (
            out["person_emp_length"] / adult_years
        ).clip(upper=1.5)

    if has("person_income") and has("cb_person_cred_hist_length"):
        out["income_per_credit_year"] = out["person_income"] / (
            1 + out["cb_person_cred_hist_length"]
        )

    if has("person_income"):
        out["log_income"] = np.log1p(out["person_income"].clip(lower=0))

    if has("cb_person_default_on_file"):
        out["has_prior_default"] = (
            out["cb_person_default_on_file"]
            .astype("object")
            .map(lambda v: 1 if isinstance(v, str) and v.strip().upper() == "Y" else 0)
        )

    if has("person_age"):
        out["age_group"] = pd.cut(
            out["person_age"],
            bins=list(config.AGE_BINS),
            labels=list(config.AGE_LABELS),
        ).astype("object")

    if has("person_income"):
        out["income_group"] = pd.cut(
            out["person_income"],
            bins=list(config.INCOME_BINS),
            labels=list(config.INCOME_LABELS),
        ).astype("object")

    if has("loan_amnt"):
        out["loan_amount_group"] = pd.cut(
            out["loan_amnt"],
            bins=list(config.LOAN_AMOUNT_BINS),
            labels=list(config.LOAN_AMOUNT_LABELS),
        ).astype("object")

    return out


def get_feature_lists(
    df: pd.DataFrame, target: str, drop: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Split the model input columns into (numeric, categorical).

    ``target`` and anything in ``drop`` are excluded from both lists.
    """
    drop = set(drop or []) | {target}
    features = [c for c in df.columns if c not in drop]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical

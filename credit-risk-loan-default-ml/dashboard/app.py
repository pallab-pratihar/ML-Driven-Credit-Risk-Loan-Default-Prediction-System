"""Streamlit dashboard for the credit-risk system.

    streamlit run dashboard/app.py

Sections: Executive Overview, Portfolio Analytics, Customer Risk Prediction,
Risk Segmentation, Model Performance, Explainability.

Everything shown here is read from artefacts produced by ``run_pipeline.py``
(scored portfolio, metrics JSON, SHAP importances) or computed live with the
saved model -- no numbers are hard-coded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `streamlit run dashboard/app.py` from the repo root
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.risk.segmentation import RiskBands, summarise_segments  # noqa: E402

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="", layout="wide")

RISK_COLOURS = {"Low Risk": "#2E8B57", "Medium Risk": "#E1A140", "High Risk": "#C0392B"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_scored_portfolio() -> pd.DataFrame | None:
    if not config.SCORED_DATA_FILE.exists():
        return None
    return pd.read_csv(config.SCORED_DATA_FILE)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if not config.METRICS_FILE.exists():
        return None
    return json.loads(config.METRICS_FILE.read_text())


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame | None:
    if not config.SHAP_IMPORTANCE_FILE.exists():
        return None
    return pd.read_csv(config.SHAP_IMPORTANCE_FILE)


@st.cache_resource(show_spinner=False)
def load_scoring_artifacts():
    from src.models.predict import load_artifacts

    return load_artifacts()


def require_artifacts() -> pd.DataFrame:
    portfolio = load_scored_portfolio()
    if portfolio is None:
        st.error("No scored portfolio found. Run `python run_pipeline.py` first.")
        st.stop()
    return portfolio


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Credit Risk Analytics")
page = st.sidebar.radio(
    "Section",
    [
        "Executive Overview",
        "Portfolio Analytics",
        "Customer Risk Prediction",
        "Risk Segmentation",
        "Model Performance",
        "Explainability",
    ],
)

portfolio = require_artifacts()
metrics = load_metrics() or {}
target = metrics.get("selected_model") and "loan_status"

st.sidebar.markdown("---")
st.sidebar.caption("Risk band thresholds (project-defined policy)")
low_threshold = st.sidebar.slider("Low risk below PD", 0.01, 0.40, config.LOW_RISK_THRESHOLD, 0.01)
high_threshold = st.sidebar.slider("High risk from PD", 0.20, 0.95, config.HIGH_RISK_THRESHOLD, 0.01)
if low_threshold >= high_threshold:
    st.sidebar.error("Low-risk cut-off must stay below the high-risk cut-off.")
    st.stop()
bands = RiskBands(low=low_threshold, high=high_threshold)

from src.risk.segmentation import assign_risk_category  # noqa: E402

portfolio = portfolio.copy()
portfolio["risk_category"] = assign_risk_category(portfolio["default_probability"], bands)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_executive_overview() -> None:
    st.title("Executive Overview")
    total = len(portfolio)
    exposure = portfolio["loan_amnt"].sum()
    default_rate = portfolio["loan_status"].mean() if "loan_status" in portfolio else float("nan")

    columns = st.columns(5)
    columns[0].metric("Customers", f"{total:,}")
    columns[1].metric("Loans", f"{total:,}")
    columns[2].metric("Observed default rate", f"{default_rate:.2%}")
    columns[3].metric("Average loan amount", f"${portfolio['loan_amnt'].mean():,.0f}")
    columns[4].metric("Total exposure", f"${exposure/1e6:,.1f}M")

    columns = st.columns(3)
    high_risk = portfolio[portfolio["risk_category"] == "High Risk"]
    columns[0].metric("High-risk customers", f"{len(high_risk):,}",
                      f"{len(high_risk)/total:.1%} of portfolio")
    columns[1].metric("Exposure in high-risk band", f"${high_risk['loan_amnt'].sum()/1e6:,.1f}M")
    columns[2].metric("Average predicted PD", f"{portfolio['default_probability'].mean():.2%}")

    left, right = st.columns(2)
    counts = portfolio["risk_category"].value_counts().reindex(bands.labels).fillna(0)
    left.plotly_chart(
        px.pie(values=counts.values, names=counts.index, hole=0.45,
               color=counts.index, color_discrete_map=RISK_COLOURS,
               title="Portfolio by risk band"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.histogram(portfolio, x="default_probability", nbins=50,
                     color="risk_category", color_discrete_map=RISK_COLOURS,
                     title="Distribution of predicted default probability"),
        use_container_width=True,
    )


def page_portfolio_analytics() -> None:
    st.title("Portfolio Analytics")
    filters = st.columns(3)
    grades = filters[0].multiselect("Loan grade", sorted(portfolio["loan_grade"].dropna().unique()))
    intents = filters[1].multiselect("Loan intent", sorted(portfolio["loan_intent"].dropna().unique()))
    ownership = filters[2].multiselect(
        "Home ownership", sorted(portfolio["person_home_ownership"].dropna().unique())
    )

    view = portfolio
    if grades:
        view = view[view["loan_grade"].isin(grades)]
    if intents:
        view = view[view["loan_intent"].isin(intents)]
    if ownership:
        view = view[view["person_home_ownership"].isin(ownership)]

    st.caption(f"{len(view):,} loans selected ({len(view)/len(portfolio):.1%} of the portfolio)")
    if view.empty:
        st.warning("No loans match the current filters.")
        return

    dimension = st.selectbox(
        "Break down default rate by",
        ["loan_grade", "loan_intent", "person_home_ownership", "income_group",
         "age_group", "loan_amount_group"],
    )
    grouped = (
        view.groupby(dimension, observed=True)
        .agg(loans=("loan_amnt", "size"),
             observed_default_rate=("loan_status", "mean"),
             avg_predicted_pd=("default_probability", "mean"),
             exposure=("loan_amnt", "sum"))
        .reset_index()
        .sort_values("observed_default_rate", ascending=False)
    )
    st.plotly_chart(
        px.bar(grouped, x=dimension, y=["observed_default_rate", "avg_predicted_pd"],
               barmode="group", title=f"Default rate by {dimension}"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.scatter(view.sample(min(4000, len(view)), random_state=config.RANDOM_STATE),
                   x="person_income", y="loan_amnt", color="risk_category",
                   color_discrete_map=RISK_COLOURS, opacity=0.6,
                   log_x=True, title="Income vs loan amount by risk band"),
        use_container_width=True,
    )
    st.dataframe(grouped.round(4), use_container_width=True)


def page_customer_prediction() -> None:
    st.title("Customer Risk Prediction")
    st.caption("Scores a single application with the saved model pipeline.")

    left, middle, right = st.columns(3)
    age = left.number_input("Age", 18, 100, 30)
    income = left.number_input("Annual income", 4000, 2_000_000, 55_000, step=1000)
    emp_length = left.number_input("Employment length (years)", 0.0, 60.0, 4.0, step=0.5)
    home = middle.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    intent = middle.selectbox(
        "Loan intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    grade = middle.selectbox("Loan grade", list("ABCDEFG"), index=2)
    amount = right.number_input("Loan amount", 500, 100_000, 12_000, step=500)
    rate = right.number_input("Interest rate (%)", 0.0, 40.0, 12.0, step=0.1)
    prior_default = right.selectbox("Prior default on bureau file", ["N", "Y"])
    history = right.number_input("Credit history length (years)", 0, 60, 6)

    if not st.button("Score application", type="primary"):
        return

    from src.models.predict import score_records

    record = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": emp_length,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": rate,
        "cb_person_default_on_file": prior_default,
        "cb_person_cred_hist_length": history,
    }
    result = score_records(record, bands=bands)[0]

    columns = st.columns(3)
    columns[0].metric("Probability of default", f"{result['default_probability']:.2%}")
    columns[1].metric("Probability of repayment", f"{result['non_default_probability']:.2%}")
    columns[2].metric("Risk category", result["risk_category"])
    st.progress(min(result["default_probability"], 1.0))
    st.caption(
        f"Decision at the tuned threshold ({result['decision_threshold']:.2f}): "
        + ("**flag as likely default**" if result["prediction"] else "**accept**")
    )

    with st.spinner("Computing SHAP explanation..."):
        try:
            from src.features.engineering import add_credit_risk_features
            from src.models.explain import build_explainer, explain_customer, narrate_contributions
            from src.models.predict import prepare_input

            pipeline, _ = load_scoring_artifacts()
            customer_frame = add_credit_risk_features(prepare_input(record))
            background = add_credit_risk_features(
                portfolio.drop(columns=["default_probability", "risk_category", "loan_status"],
                               errors="ignore").sample(200, random_state=config.RANDOM_STATE)
            )
            explainer, _ = build_explainer(pipeline, background)
            explanation = explain_customer(pipeline, explainer, customer_frame)
        except Exception as exc:  # pragma: no cover - dashboard convenience
            st.info(f"Explanation unavailable: {exc}")
            return

    st.subheader("Why this score")
    left, right = st.columns(2)
    left.markdown("**Pushing risk up**")
    left.dataframe(pd.DataFrame(explanation["increases_risk"]), hide_index=True)
    right.markdown("**Pushing risk down**")
    right.dataframe(pd.DataFrame(explanation["decreases_risk"]), hide_index=True)
    for line in narrate_contributions(explanation):
        st.write("- " + line)


def page_risk_segmentation() -> None:
    st.title("Risk Segmentation")
    st.caption(
        "Bands are project-defined policy thresholds, adjustable in the sidebar: "
        + ", ".join(f"{k} = {v}" for k, v in bands.describe().items())
    )
    summary = summarise_segments(portfolio, bands, target="loan_status")
    st.dataframe(summary, use_container_width=True)

    reset = summary.reset_index()
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(reset, x="risk_category", y="customers", color="risk_category",
               color_discrete_map=RISK_COLOURS, title="Customers per band"),
        use_container_width=True,
    )
    if "actual_default_rate" in reset:
        right.plotly_chart(
            px.bar(reset, x="risk_category",
                   y=["avg_predicted_pd", "actual_default_rate"], barmode="group",
                   title="Predicted PD vs observed default rate"),
            use_container_width=True,
        )
    st.plotly_chart(
        px.bar(reset, x="risk_category", y="total_exposure", color="risk_category",
               color_discrete_map=RISK_COLOURS, title="Exposure per band"),
        use_container_width=True,
    )


def page_model_performance() -> None:
    st.title("Model Performance")
    if not metrics:
        st.warning("No metrics file found. Run `python run_pipeline.py` first.")
        return

    st.metric("Selected model", metrics["selected_model"])
    columns = st.columns(3)
    columns[0].metric("Decision threshold", f"{metrics['decision_threshold']:.2f}")
    columns[1].metric("Imbalance handling", metrics["imbalance_strategy_per_model"][metrics["selected_model"]])
    columns[2].metric("Test set size", f"{metrics['split_sizes']['test']:,}")

    st.subheader("Test-set comparison")
    st.dataframe(pd.DataFrame(metrics["test_metrics"]).T.round(4), use_container_width=True)
    st.subheader("Validation-set comparison (used for model selection)")
    st.dataframe(pd.DataFrame(metrics["validation_metrics"]).T.round(4), use_container_width=True)

    policy = metrics.get("high_recall_policy")
    if policy:
        st.subheader("Alternative high-recall credit policy")
        st.caption(
            f"Threshold {policy['threshold']:.2f}, chosen on validation with a "
            f"recall floor of {policy['recall_floor']:.0%}."
        )
        st.dataframe(pd.DataFrame({"test": policy["test_metrics"]}).T.round(4),
                     use_container_width=True)

    for name, caption in [
        ("roc_pr_curves_test.png", "ROC and precision-recall curves (test set)"),
        ("confusion_matrix.png", "Confusion matrix at the tuned threshold"),
        ("calibration_curve.png", "Calibration (validation set)"),
        ("pd_distribution.png", "Predicted PD by actual outcome"),
    ]:
        path = config.FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=caption)


def page_explainability() -> None:
    st.title("Explainability")
    importance = load_shap_importance()
    if importance is None:
        st.warning("No SHAP output found. Run `python run_pipeline.py` first.")
        return
    top = importance.head(15).sort_values("mean_abs_shap")
    st.plotly_chart(
        px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
               title="Global feature importance (mean |SHAP|)"),
        use_container_width=True,
    )
    st.caption(
        "SHAP values are additive log-odds contributions computed on a test-set "
        "sample; positive values push a customer towards default."
    )
    summary_plot = config.FIGURES_DIR / "shap_summary.png"
    if summary_plot.exists():
        st.image(str(summary_plot), caption="SHAP summary (beeswarm)")
    st.dataframe(importance.round(4), use_container_width=True)


PAGES = {
    "Executive Overview": page_executive_overview,
    "Portfolio Analytics": page_portfolio_analytics,
    "Customer Risk Prediction": page_customer_prediction,
    "Risk Segmentation": page_risk_segmentation,
    "Model Performance": page_model_performance,
    "Explainability": page_explainability,
}

PAGES[page]()

"""End-to-end reproducible pipeline for the credit-risk project.

    python run_pipeline.py                  # everything
    python run_pipeline.py --steps clean features model
    python run_pipeline.py --skip-shap      # faster run without SHAP

Stages: clean -> features -> eda -> sql -> model -> shap -> score.
Every stage writes its artefacts under data/processed/, models/ or reports/,
so the notebooks and the serving layer consume exactly what was produced here.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.analytics import runner as sql_runner
from src.data import preprocessing
from src.evaluation import metrics as metrics_mod
from src.features.engineering import FEATURE_DOCS, add_credit_risk_features, get_feature_lists
from src.models import explain as explain_mod
from src.models import train as train_mod
from src.risk.segmentation import RiskBands, segment_portfolio, summarise_segments

LOGGER = logging.getLogger("pipeline")
ALL_STEPS = ("clean", "features", "eda", "sql", "model", "shap", "score")

# Columns excluded from the feature matrix (an engineered column encodes them).
DROP_FROM_FEATURES = list(train_mod.REDUNDANT_COLUMNS)


def _configure_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def step_clean() -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Load, profile and clean the raw data."""
    raw = preprocessing.load_raw_data()
    target = preprocessing.detect_target(raw)
    LOGGER.info("Loaded %s rows x %s columns; target = %s", *raw.shape, target)

    profile = preprocessing.profile_dataframe(raw, target)
    clean, report = preprocessing.clean_data(raw, target)
    LOGGER.info(
        "Cleaned: -%s duplicates, invalid values nulled: %s",
        report["duplicates_removed"],
        report["invalid_values_nulled"],
    )
    preprocessing.save_dataframe(clean, config.CLEAN_DATA_FILE)
    (config.REPORTS_DIR / "data_profile.json").write_text(
        json.dumps({"raw_profile": profile, "cleaning_report": report}, indent=2)
    )
    return clean, target, report


def step_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise credit-risk features."""
    featured = add_credit_risk_features(clean)
    new_columns = [c for c in featured.columns if c not in clean.columns]
    LOGGER.info("Engineered %s features: %s", len(new_columns), ", ".join(new_columns))
    preprocessing.save_dataframe(featured, config.FEATURED_DATA_FILE)
    return featured


def step_eda(featured: pd.DataFrame, target: str) -> None:
    """Generate the EDA figures used in the report and the README."""
    from src import eda

    eda.plot_target_distribution(featured, target)
    eda.plot_numeric_distributions(
        featured,
        ["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income",
         "person_age", "person_emp_length"],
        target,
    )
    eda.plot_default_rate_bars(
        featured,
        ["loan_grade", "loan_intent", "person_home_ownership", "age_group",
         "income_group", "loan_amount_group"],
        target,
    )
    numeric_cols = [
        c for c in featured.select_dtypes(include=np.number).columns if c != target
    ]
    eda.plot_correlation_heatmap(featured, numeric_cols)
    eda.plot_ratio_vs_default(featured, "loan_percent_income", target)
    LOGGER.info("EDA figures written to %s", config.FIGURES_DIR)


def step_sql(featured: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the DuckDB analytics layer."""
    results = sql_runner.run_queries(featured)
    sql_runner.save_report(results)
    LOGGER.info("Ran %s SQL queries -> %s", len(results), config.SQL_RESULTS_FILE)
    return results


def step_model(featured: pd.DataFrame, target: str) -> dict[str, Any]:
    """Split, train the three candidates, select one, and evaluate on test."""
    splits = train_mod.split_data(featured, target)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    LOGGER.info(
        "Split sizes -> train %s | val %s | test %s (default rate %.4f / %.4f / %.4f)",
        len(X_train), len(X_val), len(X_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )

    numeric, categorical = get_feature_lists(featured, target, drop=DROP_FROM_FEATURES)
    LOGGER.info("%s numeric and %s categorical features", len(numeric), len(categorical))

    scale_pos_weight = train_mod.compute_scale_pos_weight(y_train)

    # ---- class-imbalance study (decided on the VALIDATION set) -----------
    # Each model is fitted twice -- with and without minority-class
    # re-weighting -- and the better variant per model is kept. SMOTE is not
    # used: at a ~22% default rate the minority class is well represented, and
    # synthetic borrowers would distort the probabilities the risk bands rely on.
    imbalance_study: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    chosen_strategy: dict[str, str] = {}
    for strategy, weighting in (("class-weighted", True), ("unweighted", False)):
        candidates = train_mod.build_models(
            numeric, categorical, scale_pos_weight, class_weighting=weighting
        )
        for name, pipeline in train_mod.fit_models(candidates, X_train, y_train).items():
            scores = metrics_mod.evaluate_predictions(
                y_val, train_mod.predict_proba(pipeline, X_val)
            )
            imbalance_study.setdefault(name, {})[strategy] = scores
            incumbent = chosen_strategy.get(name)
            better = incumbent is None or (
                scores["PR-AUC"],
                -scores["Brier"],
            ) > (
                imbalance_study[name][incumbent]["PR-AUC"],
                -imbalance_study[name][incumbent]["Brier"],
            )
            if better:
                chosen_strategy[name] = strategy
                fitted[name] = pipeline
    LOGGER.info("Imbalance strategy kept per model: %s", chosen_strategy)
    pd.DataFrame(
        {(m, s): v for m, res in imbalance_study.items() for s, v in res.items()}
    ).T.round(4).to_csv(config.REPORTS_DIR / "imbalance_study.csv")

    # ---- model selection on the VALIDATION set only ----------------------
    val_proba = {n: train_mod.predict_proba(p, X_val) for n, p in fitted.items()}
    val_metrics = {
        n: metrics_mod.evaluate_predictions(y_val, p) for n, p in val_proba.items()
    }
    val_table = metrics_mod.comparison_table(val_metrics)
    LOGGER.info("Validation comparison:\n%s", val_table.to_string())

    best_name = str(val_table.index[0])  # ranked by PR-AUC
    best_model = fitted[best_name]
    threshold, threshold_metrics = metrics_mod.tune_threshold(
        y_val, val_proba[best_name], objective="f1"
    )
    LOGGER.info("Selected %s; tuned threshold = %.2f", best_name, threshold)

    # Alternative credit policy: accept more false alarms to catch more
    # defaults. Chosen on validation, reported on test, never used to select
    # the model itself.
    recall_floor = 0.85
    policy_threshold, policy_val_metrics = metrics_mod.tune_threshold(
        y_val, val_proba[best_name],
        objective="max_precision_at_recall", min_recall=recall_floor,
    )
    LOGGER.info(
        "High-recall policy threshold = %.2f (validation recall %.3f, precision %.3f)",
        policy_threshold, policy_val_metrics["Recall"], policy_val_metrics["Precision"],
    )

    metrics_mod.plot_roc_pr_curves(y_val, val_proba, filename="roc_pr_curves_validation.png")
    metrics_mod.plot_calibration(y_val, val_proba)

    # ---- single, final evaluation on the TEST set ------------------------
    test_proba = {n: train_mod.predict_proba(p, X_test) for n, p in fitted.items()}
    test_metrics = {
        n: metrics_mod.evaluate_predictions(
            y_test, p, threshold if n == best_name else config.DEFAULT_DECISION_THRESHOLD
        )
        for n, p in test_proba.items()
    }
    test_table = metrics_mod.comparison_table(test_metrics)
    LOGGER.info("Test comparison:\n%s", test_table.to_string())

    best_proba = test_proba[best_name]
    policy_test_metrics = metrics_mod.evaluate_predictions(y_test, best_proba, policy_threshold)
    report_text = metrics_mod.classification_text_report(y_test, best_proba, threshold)
    metrics_mod.plot_confusion_matrix(
        y_test, best_proba, threshold, title=f"{best_name} - test set"
    )
    metrics_mod.plot_roc_pr_curves(y_test, test_proba, filename="roc_pr_curves_test.png")
    metrics_mod.plot_probability_distribution(y_test, best_proba)

    # ---- risk segmentation on the test portfolio -------------------------
    bands = RiskBands()
    scored_test = segment_portfolio(
        X_test.assign(**{target: y_test}), best_proba, bands, target=target
    )
    segment_table = summarise_segments(scored_test, bands, target=target)
    LOGGER.info("Test-set risk segmentation:\n%s", segment_table.to_string())

    # ---- persist artefacts ------------------------------------------------
    train_mod.save_model(best_model)
    raw_input_columns = [
        c for c in preprocessing.load_raw_data().columns if c != target
    ]
    feature_config = {
        "target": target,
        "selected_model": best_name,
        "imbalance_strategy": chosen_strategy[best_name],
        "decision_threshold": threshold,
        "threshold_objective": "max F1 for the default class on the validation set",
        "risk_bands": {"low": bands.low, "high": bands.high, "labels": list(bands.labels)},
        "raw_input_columns": raw_input_columns,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "engineered_feature_docs": FEATURE_DOCS,
        "categorical_levels": {
            c: sorted(map(str, featured[c].dropna().unique())) for c in categorical
        },
        "random_state": config.RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    config.FEATURE_CONFIG_FILE.write_text(json.dumps(feature_config, indent=2))

    test_table.to_csv(config.MODEL_COMPARISON_FILE)
    segment_table.to_csv(config.REPORTS_DIR / "risk_segmentation_test.csv")
    (config.REPORTS_DIR / "classification_report.txt").write_text(report_text)

    metrics_mod.summarise_run(
        {
            "selected_model": best_name,
            "imbalance_strategy_per_model": chosen_strategy,
            "imbalance_study_validation": imbalance_study,
            "decision_threshold": threshold,
            "validation_metrics": val_metrics,
            "validation_metrics_at_tuned_threshold": threshold_metrics,
            "high_recall_policy": {
                "recall_floor": recall_floor,
                "threshold": policy_threshold,
                "validation_metrics": policy_val_metrics,
                "test_metrics": policy_test_metrics,
            },
            "test_metrics": test_metrics,
            "split_sizes": {
                "train": int(len(X_train)),
                "validation": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "class_balance": {
                "train_default_rate": float(y_train.mean()),
                "validation_default_rate": float(y_val.mean()),
                "test_default_rate": float(y_test.mean()),
                "scale_pos_weight": scale_pos_weight,
            },
            "risk_segmentation_test": json.loads(segment_table.to_json(orient="index")),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    LOGGER.info("Artefacts written to %s and %s", config.MODELS_DIR, config.REPORTS_DIR)

    return {
        "fitted": fitted,
        "best_name": best_name,
        "best_model": best_model,
        "splits": splits,
        "threshold": threshold,
        "test_table": test_table,
        "val_table": val_table,
        "segment_table": segment_table,
    }


def step_shap(model_output: dict[str, Any], sample_size: int = 2000) -> None:
    """Compute SHAP values for the selected model on a test-set sample."""
    best_model = model_output["best_model"]
    X_test = model_output["splits"]["X_test"]
    sample = X_test.sample(
        n=min(sample_size, len(X_test)), random_state=config.RANDOM_STATE
    )
    explainer, transformed = explain_mod.build_explainer(best_model, sample)
    shap_values = explain_mod.compute_shap_values(explainer, transformed)
    importance = explain_mod.global_importance(shap_values, transformed)
    importance.to_csv(config.SHAP_IMPORTANCE_FILE, index=False)
    explain_mod.save_summary_plots(shap_values, transformed)
    LOGGER.info("Top SHAP drivers:\n%s", importance.head(10).to_string(index=False))


def step_score(featured: pd.DataFrame, target: str) -> None:
    """Score the full portfolio with the saved artefacts (dashboard input)."""
    from src.models.predict import score_dataframe

    scored = score_dataframe(featured.drop(columns=[target]))
    scored[target] = featured[target].to_numpy()
    preprocessing.save_dataframe(scored, config.SCORED_DATA_FILE)
    LOGGER.info(
        "Scored portfolio saved (%s rows, mean PD %.4f)",
        len(scored),
        scored["default_probability"].mean(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(steps: list[str], skip_shap: bool = False) -> None:
    """Run the requested pipeline stages in order."""
    config.ensure_directories()
    np.random.seed(config.RANDOM_STATE)

    clean, target, _ = step_clean()
    featured = step_features(clean) if "features" in steps else clean

    if "eda" in steps:
        step_eda(featured, target)
    if "sql" in steps:
        step_sql(featured)

    model_output: dict[str, Any] | None = None
    if "model" in steps:
        model_output = step_model(featured, target)
    if "shap" in steps and model_output and not skip_shap:
        step_shap(model_output)
    if "score" in steps and config.MODEL_FILE.exists():
        step_score(featured, target)

    LOGGER.info("Pipeline finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit-risk pipeline runner")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=list(ALL_STEPS),
        help="subset of stages to run (order is fixed)",
    )
    parser.add_argument("--skip-shap", action="store_true", help="skip SHAP computation")
    parser.add_argument("--quiet", action="store_true", help="reduce log output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure_logging(verbose=not args.quiet)
    main(steps=list(args.steps), skip_shap=args.skip_shap)

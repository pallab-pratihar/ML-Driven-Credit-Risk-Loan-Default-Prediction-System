"""SHAP explainability for the selected credit-risk model.

Every explanation returned here is derived from real SHAP values computed on
the fitted pipeline; nothing is hand-written. SHAP values are additive
contributions in log-odds space, so a positive value pushes the customer
towards default and a negative value away from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src import config
from src.models.train import get_transformed_feature_names


def _sanitise(name: str) -> str:
    """XGBoost rejects feature names containing ``[``, ``]`` or ``<``."""
    return (
        str(name)
        .replace("<", "lt_")
        .replace(">", "gt_")
        .replace("[", "(")
        .replace("]", ")")
    )


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a named, dense frame."""
    matrix = pipeline.named_steps["preprocessor"].transform(X)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    columns = [_sanitise(c) for c in get_transformed_feature_names(pipeline)]
    return pd.DataFrame(matrix, columns=columns, index=X.index)


def build_explainer(pipeline: Pipeline, X_background: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Create a SHAP explainer for the estimator inside ``pipeline``.

    Tree ensembles use the exact TreeExplainer; anything else falls back to the
    model-agnostic Explainer with a small background sample.
    """
    model = pipeline.named_steps["classifier"]
    transformed = transform_features(pipeline, X_background)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:  # pragma: no cover - non-tree fallback
        explainer = shap.Explainer(model, transformed)
    return explainer, transformed


def compute_shap_values(explainer: Any, transformed: pd.DataFrame) -> np.ndarray:
    """Return a 2-D array of SHAP values for the positive class."""
    values = explainer.shap_values(transformed)
    values = np.asarray(values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, 1]
    return values


def global_importance(shap_values: np.ndarray, transformed: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature -- the global importance ranking."""
    importance = pd.DataFrame(
        {
            "feature": transformed.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_summary_plots(
    shap_values: np.ndarray,
    transformed: pd.DataFrame,
    out_dir: Path = config.FIGURES_DIR,
    max_display: int = 15,
) -> dict[str, Path]:
    """Save the beeswarm summary plot and the bar importance plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    plt.figure()
    shap.summary_plot(shap_values, transformed, max_display=max_display, show=False)
    paths["beeswarm"] = out_dir / "shap_summary.png"
    plt.savefig(paths["beeswarm"], dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values, transformed, plot_type="bar", max_display=max_display, show=False
    )
    paths["bar"] = out_dir / "shap_importance_bar.png"
    plt.savefig(paths["bar"], dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return paths


def explain_customer(
    pipeline: Pipeline,
    explainer: Any,
    customer: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, list[dict[str, float | str]]]:
    """Explain a single prediction.

    Returns the ``top_n`` features pushing the PD up and the ``top_n`` pushing
    it down, each with its SHAP value and the customer's feature value.
    """
    if len(customer) != 1:
        raise ValueError("explain_customer expects exactly one row")
    transformed = transform_features(pipeline, customer)
    values = compute_shap_values(explainer, transformed)[0]

    contributions = pd.DataFrame(
        {
            "feature": transformed.columns,
            "shap_value": values,
            "feature_value": transformed.iloc[0].to_numpy(),
        }
    ).sort_values("shap_value", ascending=False)

    increases = contributions.head(top_n)
    decreases = contributions.tail(top_n).sort_values("shap_value")
    to_records = lambda frame: [  # noqa: E731 - concise local formatter
        {
            "feature": row.feature,
            "shap_value": round(float(row.shap_value), 4),
            "feature_value": round(float(row.feature_value), 4),
        }
        for row in frame.itertuples()
    ]
    return {
        "increases_risk": to_records(increases),
        "decreases_risk": to_records(decreases),
    }


def narrate_contributions(explanation: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Turn SHAP contributions into plain sentences (values stay verbatim)."""
    lines = [
        f"{item['feature']} increased predicted default risk "
        f"(SHAP {item['shap_value']:+.3f})"
        for item in explanation["increases_risk"]
        if item["shap_value"] > 0
    ]
    lines += [
        f"{item['feature']} reduced predicted default risk "
        f"(SHAP {item['shap_value']:+.3f})"
        for item in explanation["decreases_risk"]
        if item["shap_value"] < 0
    ]
    return lines

"""Classification metrics, threshold tuning and evaluation plots.

Credit-risk models are judged on ranking quality (ROC-AUC, PR-AUC) and on the
recall/precision trade-off for the *default* class, not on accuracy: with a
~22% default rate, predicting "nobody defaults" already scores ~78% accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks override with %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

METRIC_COLUMNS: list[str] = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = config.DEFAULT_DECISION_THRESHOLD,
) -> dict[str, float]:
    """Compute the full metric set for one model at one decision threshold.

    Precision/recall/F1 are reported for the positive (default) class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba)),
        "Brier": float(brier_score_loss(y_true, y_proba)),
    }


def tune_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    objective: str = "f1",
    min_recall: float = 0.60,
) -> tuple[float, dict[str, float]]:
    """Pick a decision threshold on the **validation** set.

    Args:
        y_true: validation labels.
        y_proba: validation probabilities from the fitted model.
        objective: ``"f1"`` maximises F1 for the default class;
            ``"max_precision_at_recall"`` maximises precision subject to
            recall >= ``min_recall`` (a typical credit-policy formulation).
        min_recall: recall floor used by the second objective.

    Returns:
        (chosen threshold, metrics at that threshold).
    """
    y_true = np.asarray(y_true).astype(int)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [evaluate_predictions(y_true, y_proba, t) for t in grid]
    table = pd.DataFrame(rows)

    if objective == "f1":
        best = table.loc[table["F1"].idxmax()]
    elif objective == "max_precision_at_recall":
        eligible = table[table["Recall"] >= min_recall]
        if eligible.empty:
            raise ValueError(f"No threshold reaches recall >= {min_recall}")
        best = eligible.loc[eligible["Precision"].idxmax()]
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    threshold = float(best["threshold"])
    return threshold, best.to_dict()


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{model_name: metric_dict}`` into the report comparison table."""
    table = pd.DataFrame(results).T
    keep = [c for c in METRIC_COLUMNS if c in table.columns]
    return table[keep].round(4).sort_values("PR-AUC", ascending=False)


def classification_text_report(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float
) -> str:
    """Return sklearn's classification report as text."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return classification_report(
        np.asarray(y_true).astype(int),
        y_pred,
        target_names=["Non-default (0)", "Default (1)"],
        digits=4,
        zero_division=0,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    title: str = "Confusion matrix",
    out_dir: Path = config.FIGURES_DIR,
    filename: str = "confusion_matrix.png",
) -> Path:
    """Save an annotated confusion matrix at the given threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred)
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    ax.imshow(cm, cmap="Blues")
    labels = ["Non-default", "Default"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} (threshold = {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    return _save(fig, filename, out_dir)


def plot_roc_pr_curves(
    y_true: np.ndarray | pd.Series,
    proba_by_model: dict[str, np.ndarray],
    out_dir: Path = config.FIGURES_DIR,
    filename: str = "roc_pr_curves.png",
) -> Path:
    """Overlay ROC and precision-recall curves for several models."""
    y_true = np.asarray(y_true).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=config.FIGSIZE_WIDE)
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
        precision, recall, _ = precision_recall_curve(y_true, proba)
        axes[1].plot(
            recall,
            precision,
            label=f"{name} (AP={average_precision_score(y_true, proba):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    baseline = y_true.mean()
    axes[1].axhline(baseline, ls="--", c="k", lw=1, label=f"Base rate ({baseline:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, filename, out_dir)


def plot_calibration(
    y_true: np.ndarray | pd.Series,
    proba_by_model: dict[str, np.ndarray],
    n_bins: int = 10,
    out_dir: Path = config.FIGURES_DIR,
    filename: str = "calibration_curve.png",
) -> Path:
    """Reliability diagram: are predicted PDs close to observed default rates?"""
    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    for name, proba in proba_by_model.items():
        observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, "o-", label=f"{name} (Brier={brier_score_loss(y_true, proba):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        title="Calibration (validation set)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, filename, out_dir)


def plot_probability_distribution(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    out_dir: Path = config.FIGURES_DIR,
    filename: str = "pd_distribution.png",
) -> Path:
    """Predicted-PD histograms split by the observed outcome."""
    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    bins = np.linspace(0, 1, 41)
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.6, label="Actual non-default", density=True)
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.6, label="Actual default", density=True)
    ax.set(xlabel="Predicted probability of default", ylabel="Density",
           title="Predicted PD by actual outcome (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, filename, out_dir)


def metrics_to_markdown(table: pd.DataFrame) -> str:
    """Render the comparison table as GitHub-flavoured markdown."""
    header = "| Model | " + " | ".join(table.columns) + " |"
    sep = "|" + "---|" * (len(table.columns) + 1)
    lines = [header, sep]
    for name, row in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    return "\n".join(lines)


def summarise_run(results: dict[str, Any], path: Path = config.METRICS_FILE) -> Path:
    """Persist a JSON record of an evaluation run."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float))
    return path

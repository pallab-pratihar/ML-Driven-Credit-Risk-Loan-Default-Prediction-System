"""Exploratory analysis helpers shared by notebook 02 and ``run_pipeline.py``.

Each function answers one business question and returns either a tidy frame or
the path of the saved figure, so the same code produces the notebook output and
the report artefacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig: plt.Figure, filename: str, out_dir: Path = config.FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def default_rate_by(
    df: pd.DataFrame, column: str, target: str, min_count: int = 30
) -> pd.DataFrame:
    """Default rate, volume and average loan size for each level of ``column``."""
    grouped = df.groupby(column, observed=True)
    table = pd.DataFrame(
        {
            "customers": grouped.size(),
            "default_rate": grouped[target].mean(),
        }
    )
    if "loan_amnt" in df.columns:
        table["avg_loan_amount"] = grouped["loan_amnt"].mean()
    if "person_income" in df.columns:
        table["avg_income"] = grouped["person_income"].mean()
    table = table[table["customers"] >= min_count]
    return table.sort_values("default_rate", ascending=False).round(4)


def numeric_summary_by_target(
    df: pd.DataFrame, columns: list[str], target: str
) -> pd.DataFrame:
    """Mean of each numeric feature split by outcome, plus the relative gap."""
    grouped = df.groupby(target)[columns].mean().T
    grouped.columns = ["non_default_mean", "default_mean"]
    grouped["relative_gap_pct"] = (
        (grouped["default_mean"] - grouped["non_default_mean"])
        / grouped["non_default_mean"].replace(0, np.nan)
        * 100
    )
    return grouped.round(3)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, target: str) -> Path:
    """Class balance of the target."""
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    ax.bar(["Non-default (0)", "Default (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}\n({value / len(df):.1%})", ha="center", va="bottom")
    ax.set(ylabel="Customers", title="Target distribution")
    ax.margins(y=0.15)
    return _save(fig, "target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str], target: str) -> Path:
    """Overlaid distributions of key numeric drivers by outcome."""
    columns = [c for c in columns if c in df.columns][:6]
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    for ax, column in zip(np.ravel(axes), columns):
        upper = df[column].quantile(0.99)
        subset = df[df[column] <= upper]
        sns.kdeplot(data=subset, x=column, hue=target, common_norm=False, fill=True, ax=ax)
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("")
    for ax in np.ravel(axes)[len(columns):]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions by outcome (99th-percentile clipped)")
    fig.tight_layout()
    return _save(fig, "numeric_distributions.png")


def plot_default_rate_bars(df: pd.DataFrame, columns: list[str], target: str) -> Path:
    """Default rate across the main categorical / banded variables."""
    columns = [c for c in columns if c in df.columns][:6]
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.4 * n_rows))
    overall = df[target].mean()
    for ax, column in zip(np.ravel(axes), columns):
        rates = df.groupby(column, observed=True)[target].mean().sort_values()
        ax.bar(rates.index.astype(str), rates.values, color="#4C72B0")
        ax.axhline(overall, color="#C44E52", ls="--", lw=1)
        ax.set_title(f"Default rate by {column}", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylim(0, max(0.05, rates.max() * 1.2))
    for ax in np.ravel(axes)[len(columns):]:
        ax.axis("off")
    fig.suptitle(f"Dashed line = portfolio default rate ({overall:.1%})")
    fig.tight_layout()
    return _save(fig, "default_rate_by_segment.png")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None) -> Path:
    """Pearson correlation between numeric features (collinearity check)."""
    numeric = df[columns] if columns else df.select_dtypes(include=np.number)
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.0 * len(corr) + 2, 0.8 * len(corr) + 2))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=False, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation matrix (numeric features)")
    return _save(fig, "correlation_heatmap.png")


def plot_ratio_vs_default(df: pd.DataFrame, ratio_column: str, target: str,
                          n_bins: int = 10) -> Path:
    """Default rate across deciles of an affordability ratio."""
    working = df[[ratio_column, target]].dropna()
    working["bucket"] = pd.qcut(working[ratio_column], q=n_bins, duplicates="drop")
    rates = working.groupby("bucket", observed=True)[target].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    ax.bar(range(len(rates)), rates["mean"].values, color="#4C72B0")
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([str(i) for i in rates.index], rotation=45, ha="right", fontsize=8)
    ax.set(xlabel=f"{ratio_column} decile", ylabel="Default rate",
           title=f"Default rate by {ratio_column}")
    fig.tight_layout()
    return _save(fig, f"default_rate_by_{ratio_column}.png")

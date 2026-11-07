"""Risk segmentation on top of the predicted probability of default.

The bands below are **project-defined policy thresholds**, not statistical
facts. A real lender sets them from its risk appetite, expected loss-given-
default and target approval rate, then back-tests them. They are therefore
configurable everywhere (config module, dataclass argument, API, dashboard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class RiskBands:
    """Cut-offs that map a predicted PD onto a risk category."""

    low: float = config.LOW_RISK_THRESHOLD
    high: float = config.HIGH_RISK_THRESHOLD
    labels: tuple[str, str, str] = config.RISK_LABELS

    def __post_init__(self) -> None:
        if not 0 < self.low < self.high < 1:
            raise ValueError(
                f"Require 0 < low < high < 1; got low={self.low}, high={self.high}"
            )

    def describe(self) -> dict[str, str]:
        """Human-readable definition of each band."""
        return {
            self.labels[0]: f"PD < {self.low:.0%}",
            self.labels[1]: f"{self.low:.0%} <= PD < {self.high:.0%}",
            self.labels[2]: f"PD >= {self.high:.0%}",
        }


def assign_risk_category(
    probabilities: np.ndarray | pd.Series | float, bands: RiskBands | None = None
) -> np.ndarray | str:
    """Map probabilities of default to risk labels.

    A scalar input returns a single label; an array input returns an array.
    """
    bands = bands or RiskBands()
    scalar_input = np.isscalar(probabilities)
    proba = np.atleast_1d(np.asarray(probabilities, dtype=float))
    if np.any((proba < 0) | (proba > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")

    categories = np.full(proba.shape, bands.labels[1], dtype=object)
    categories[proba < bands.low] = bands.labels[0]
    categories[proba >= bands.high] = bands.labels[2]
    return categories[0] if scalar_input else categories


def segment_portfolio(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    bands: RiskBands | None = None,
    target: str | None = None,
    amount_column: str = "loan_amnt",
) -> pd.DataFrame:
    """Attach PD and risk category to a customer table.

    Args:
        df: customer-level frame.
        probabilities: predicted PD, aligned with ``df``.
        bands: risk cut-offs.
        target: optional observed-outcome column, kept for back-testing.
        amount_column: exposure column used in the summary.

    Returns:
        A copy of ``df`` with ``default_probability`` and ``risk_category``.
    """
    bands = bands or RiskBands()
    if len(df) != len(probabilities):
        raise ValueError("df and probabilities must have the same length")
    out = df.copy()
    out["default_probability"] = np.asarray(probabilities, dtype=float)
    out["risk_category"] = assign_risk_category(out["default_probability"], bands)
    if target and target in out.columns:
        out[target] = out[target].astype(int)
    _ = amount_column  # kept for signature symmetry with summarise_segments
    return out


def summarise_segments(
    scored: pd.DataFrame,
    bands: RiskBands | None = None,
    target: str | None = None,
    amount_column: str = "loan_amnt",
) -> pd.DataFrame:
    """Portfolio view per risk band: volume, exposure, mean PD, actual rate."""
    bands = bands or RiskBands()
    grouped = scored.groupby("risk_category", observed=True)
    summary = pd.DataFrame(
        {
            "customers": grouped.size(),
            "share_of_portfolio": grouped.size() / len(scored),
            "avg_predicted_pd": grouped["default_probability"].mean(),
        }
    )
    if amount_column in scored.columns:
        summary["total_exposure"] = grouped[amount_column].sum()
        summary["avg_loan_amount"] = grouped[amount_column].mean()
    if target and target in scored.columns:
        summary["actual_default_rate"] = grouped[target].mean()
        if amount_column in scored.columns:
            summary["defaulted_exposure"] = grouped.apply(
                lambda g: (g[amount_column] * g[target]).sum(), include_groups=False
            )
    order = [label for label in bands.labels if label in summary.index]
    return summary.loc[order].round(4)

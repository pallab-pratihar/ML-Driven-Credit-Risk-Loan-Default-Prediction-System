"""Inference: raw customer record -> probability of default -> risk category.

The same feature-engineering function used in training is applied here, and
the persisted pipeline carries its own fitted imputers/scaler/encoder, so
training and serving cannot drift apart.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.data.preprocessing import normalise_text_columns
from src.features.engineering import add_credit_risk_features
from src.models.train import load_model
from src.risk.segmentation import RiskBands, assign_risk_category


def load_feature_config(path: Path = config.FEATURE_CONFIG_FILE) -> dict[str, Any]:
    """Read the JSON contract written by the training pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"No feature config at {path}. Run `python run_pipeline.py` first."
        )
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Pipeline, dict[str, Any]]:
    """Load (and cache) the fitted pipeline and its feature contract."""
    return load_model(), load_feature_config()


def prepare_input(records: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Turn one record or many into the model's raw input frame.

    Missing optional columns are added as NaN and imputed by the pipeline;
    ``loan_percent_income`` is derived when the caller does not supply it.
    """
    _, feature_config = load_artifacts()
    expected: list[str] = feature_config["raw_input_columns"]

    if isinstance(records, Mapping):
        records = [records]
    frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("No records supplied")

    if (
        "loan_percent_income" in expected
        and "loan_percent_income" not in frame.columns
        and {"loan_amnt", "person_income"}.issubset(frame.columns)
    ):
        income = frame["person_income"].where(frame["person_income"] > 0)
        frame["loan_percent_income"] = (frame["loan_amnt"] / income).round(4)

    for column in expected:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = normalise_text_columns(frame)

    return frame[expected]


def score_records(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    bands: RiskBands | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Score customer records.

    Returns one dict per record with the default probability, the
    complementary non-default probability, the risk band and the hard 0/1
    prediction at the tuned decision threshold.
    """
    pipeline, feature_config = load_artifacts()
    bands = bands or RiskBands(
        low=feature_config["risk_bands"]["low"],
        high=feature_config["risk_bands"]["high"],
    )
    threshold = (
        float(threshold)
        if threshold is not None
        else float(feature_config["decision_threshold"])
    )

    raw = prepare_input(records)
    engineered = add_credit_risk_features(raw)
    proba = pipeline.predict_proba(engineered)[:, 1]
    categories = assign_risk_category(proba, bands)

    return [
        {
            "default_probability": round(float(p), 4),
            "non_default_probability": round(float(1 - p), 4),
            "risk_category": str(category),
            "prediction": int(p >= threshold),
            "decision_threshold": threshold,
        }
        for p, category in zip(proba, np.atleast_1d(categories))
    ]


def score_dataframe(df: pd.DataFrame, bands: RiskBands | None = None) -> pd.DataFrame:
    """Score a whole customer table (used by the dashboard and back-testing)."""
    pipeline, feature_config = load_artifacts()
    bands = bands or RiskBands(
        low=feature_config["risk_bands"]["low"],
        high=feature_config["risk_bands"]["high"],
    )
    engineered = add_credit_risk_features(df)
    proba = pipeline.predict_proba(engineered)[:, 1]
    out = df.copy()
    out["default_probability"] = proba
    out["risk_category"] = assign_risk_category(proba, bands)
    return out

"""Tests for the credit-risk pipeline.

Run with ``pytest`` from the repository root. Tests that need the trained
artefacts are skipped automatically when ``run_pipeline.py`` has not been run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import preprocessing
from src.features.engineering import add_credit_risk_features, get_feature_lists
from src.models import train as train_mod
from src.risk.segmentation import RiskBands, assign_risk_category, summarise_segments

ARTIFACTS_READY = config.MODEL_FILE.exists() and config.FEATURE_CONFIG_FILE.exists()
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_READY, reason="Run `python run_pipeline.py` to create model artefacts"
)


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    """A small synthetic frame with the same schema as the real dataset."""
    return pd.DataFrame(
        {
            "person_age": [25, 40, 200, 33, 25],       # 200 is out of range
            "person_income": [30000, 90000, 45000, 60000, 30000],
            "person_home_ownership": ["rent ", "MORTGAGE", "OWN", "RENT", "rent "],
            "person_emp_length": [2.0, 15.0, np.nan, 5.0, 2.0],
            "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL"],
            "loan_grade": ["C", "A", "D", "B", "C"],
            "loan_amnt": [8000, 12000, 6000, 15000, 8000],
            "loan_int_rate": [13.5, 7.2, np.nan, 10.1, 13.5],
            "loan_status": [1, 0, 1, 0, 1],
            "loan_percent_income": [0.27, 0.13, 0.13, 0.25, 0.27],
            "cb_person_default_on_file": ["Y", "N", "Y", "N", "Y"],
            "cb_person_cred_hist_length": [3, 18, 5, 9, 3],
        }
    )


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
def test_raw_dataset_is_present_and_has_a_target() -> None:
    df = preprocessing.load_raw_data()
    assert len(df) > 0
    assert preprocessing.detect_target(df) == "loan_status"


def test_detect_target_raises_when_absent() -> None:
    with pytest.raises(ValueError):
        preprocessing.detect_target(pd.DataFrame({"a": [1]}))


def test_clean_data_removes_duplicates_and_nulls_impossible_values(raw_sample) -> None:
    clean, report = preprocessing.clean_data(raw_sample, "loan_status")
    assert report["duplicates_removed"] == 1
    assert len(clean) == 4
    assert clean["person_age"].isna().sum() == 1          # the 200-year-old
    assert set(clean["person_home_ownership"].dropna()) <= {"RENT", "MORTGAGE", "OWN"}
    assert clean["loan_status"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def test_feature_engineering_adds_expected_columns(raw_sample) -> None:
    featured = add_credit_risk_features(raw_sample)
    for column in ["loan_to_income_ratio", "payment_to_income", "credit_history_ratio",
                   "has_prior_default", "age_group", "income_group"]:
        assert column in featured.columns
    expected_ratio = raw_sample.loc[0, "loan_amnt"] / raw_sample.loc[0, "person_income"]
    assert featured.loc[0, "loan_to_income_ratio"] == pytest.approx(expected_ratio)
    assert featured.loc[0, "has_prior_default"] == 1


def test_feature_engineering_does_not_touch_the_target(raw_sample) -> None:
    featured = add_credit_risk_features(raw_sample)
    assert featured["loan_status"].tolist() == raw_sample["loan_status"].tolist()


def test_get_feature_lists_excludes_target(raw_sample) -> None:
    featured = add_credit_risk_features(raw_sample)
    numeric, categorical = get_feature_lists(featured, "loan_status")
    assert "loan_status" not in numeric + categorical
    assert "person_income" in numeric
    assert "loan_grade" in categorical


# ---------------------------------------------------------------------------
# Splitting and model plumbing
# ---------------------------------------------------------------------------
def test_split_is_disjoint_and_stratified() -> None:
    df = add_credit_risk_features(pd.read_csv(config.CLEAN_DATA_FILE)) if config.CLEAN_DATA_FILE.exists() \
        else add_credit_risk_features(preprocessing.clean_data(preprocessing.load_raw_data(), "loan_status")[0])
    splits = train_mod.split_data(df, "loan_status")
    train_idx = set(splits["X_train"].index)
    val_idx = set(splits["X_val"].index)
    test_idx = set(splits["X_test"].index)
    assert not (train_idx & val_idx) and not (train_idx & test_idx) and not (val_idx & test_idx)
    assert len(train_idx | val_idx | test_idx) == len(df)
    rates = [splits[key].mean() for key in ("y_train", "y_val", "y_test")]
    assert max(rates) - min(rates) < 0.01


def test_preprocessor_handles_missing_and_unseen_categories(raw_sample) -> None:
    featured = add_credit_risk_features(raw_sample)
    numeric, categorical = get_feature_lists(featured, "loan_status")
    preprocessor = train_mod.build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(featured.drop(columns=["loan_status"]))
    assert not np.isnan(transformed).any()

    unseen = featured.drop(columns=["loan_status"]).head(1).copy()
    unseen["loan_grade"] = "Z"  # category never seen in training
    assert preprocessor.transform(unseen).shape[1] == transformed.shape[1]


def test_scale_pos_weight_matches_class_ratio() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    assert train_mod.compute_scale_pos_weight(y) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Risk segmentation
# ---------------------------------------------------------------------------
def test_risk_bands_reject_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        RiskBands(low=0.6, high=0.3)


def test_assign_risk_category_boundaries() -> None:
    bands = RiskBands(low=0.10, high=0.40)
    categories = assign_risk_category(np.array([0.0, 0.099, 0.10, 0.39, 0.40, 1.0]), bands)
    assert list(categories) == [
        "Low Risk", "Low Risk", "Medium Risk", "Medium Risk", "High Risk", "High Risk"
    ]
    assert assign_risk_category(0.05, bands) == "Low Risk"


def test_summarise_segments_shares_sum_to_one() -> None:
    frame = pd.DataFrame(
        {
            "default_probability": [0.02, 0.2, 0.9, 0.5],
            "loan_amnt": [1000, 2000, 3000, 4000],
            "loan_status": [0, 0, 1, 1],
        }
    )
    frame["risk_category"] = assign_risk_category(frame["default_probability"])
    summary = summarise_segments(frame, target="loan_status")
    assert summary["share_of_portfolio"].sum() == pytest.approx(1.0)
    assert summary["customers"].sum() == len(frame)


# ---------------------------------------------------------------------------
# Trained artefacts (skipped until the pipeline has been run)
# ---------------------------------------------------------------------------
@needs_artifacts
def test_saved_model_predicts_valid_probabilities(raw_sample) -> None:
    from src.models.predict import score_records

    record = raw_sample.drop(columns=["loan_status"]).iloc[0].to_dict()
    result = score_records(record)[0]
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["default_probability"] + result["non_default_probability"] == pytest.approx(1.0)
    assert result["risk_category"] in config.RISK_LABELS
    assert result["prediction"] in (0, 1)


@needs_artifacts
def test_risk_ordering_is_monotone_for_two_contrasting_customers() -> None:
    from src.models.predict import score_records

    safe = {
        "person_age": 45, "person_income": 150_000, "person_home_ownership": "MORTGAGE",
        "person_emp_length": 15, "loan_intent": "EDUCATION", "loan_grade": "A",
        "loan_amnt": 5_000, "loan_int_rate": 6.0,
        "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 20,
    }
    risky = {**safe, "person_income": 22_000, "loan_grade": "F", "loan_amnt": 20_000,
             "loan_int_rate": 21.0, "cb_person_default_on_file": "Y",
             "person_home_ownership": "RENT", "person_emp_length": 0.5}
    scores = score_records([safe, risky])
    assert scores[0]["default_probability"] < scores[1]["default_probability"]


@needs_artifacts
def test_api_endpoints() -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["model_loaded"] is True

        payload = {
            "person_age": 29, "person_income": 55_000, "person_home_ownership": "RENT",
            "person_emp_length": 4, "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "C",
            "loan_amnt": 12_000, "loan_int_rate": 13.5,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert body["risk_category"] in config.RISK_LABELS

        bad = {**payload, "person_age": 5}  # below the minimum accepted age
        assert client.post("/predict", json=bad).status_code == 422


@needs_artifacts
def test_sql_analytics_runs() -> None:
    from src.analytics.runner import run_queries

    featured = pd.read_csv(config.FEATURED_DATA_FILE)
    results = run_queries(featured)
    assert len(results) >= 12
    overview = results["01_portfolio_overview"]
    assert int(overview.loc[0, "total_customers"]) == len(featured)
    assert 0 < float(overview.loc[0, "default_rate_pct"]) < 100

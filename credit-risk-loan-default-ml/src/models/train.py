"""Model construction, splitting and training.

Leakage controls implemented here:

* The split happens **before** any fitted transformation.
* Imputation, scaling and one-hot encoding live inside a ``Pipeline`` /
  ``ColumnTransformer`` so their parameters are learned on the training fold
  only and merely *applied* to validation and test.
* The validation set drives model selection and threshold tuning; the test set
  is touched exactly once, at the very end, for the final unbiased numbers.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config

# Raw columns that an engineered feature already encodes; keeping both would
# put perfectly collinear inputs into the logistic regression.
REDUNDANT_COLUMNS: tuple[str, ...] = ("cb_person_default_on_file",)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = config.TEST_SIZE,
    validation_size: float = config.VALIDATION_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Stratified train / validation / test split.

    The dataset carries no origination or observation date, so a chronological
    (out-of-time) split is not possible; we use a stratified random split with
    a fixed seed instead and flag the limitation in the README. Stratification
    keeps the ~22% default rate identical in all three folds.
    """
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_size,
        stratify=y_temp,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """ColumnTransformer: median-impute (+scale) numerics, one-hot categoricals.

    Args:
        scale_numeric: ``True`` for distance/gradient-sensitive models
            (logistic regression); ``False`` for tree ensembles, which are
            invariant to monotone rescaling.
    """
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
    random_state: int = config.RANDOM_STATE,
    class_weighting: bool = True,
) -> dict[str, Pipeline]:
    """Return the three candidate pipelines, each with its own preprocessing.

    Args:
        scale_pos_weight: negatives/positives ratio for XGBoost.
        class_weighting: if ``True`` the estimators re-weight the minority
            (default) class; if ``False`` they train on the natural class
            distribution. ``run_pipeline.py`` fits both variants and picks per
            model on the validation set -- re-weighting helps recall but
            distorts the predicted probabilities, which matters here because
            risk bands are defined on the PD itself.

    Imbalance is always handled *inside the estimator* rather than by
    resampling: it costs nothing, invents no synthetic borrowers and leaves
    the training distribution intact.
    """
    logistic_weight = "balanced" if class_weighting else None
    forest_weight = "balanced_subsample" if class_weighting else None
    xgb_weight = scale_pos_weight if class_weighting else 1.0

    scaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    unscaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    logistic = Pipeline(
        [
            ("preprocessor", scaled),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=logistic_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            ("preprocessor", unscaled),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight=forest_weight,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    xgboost = Pipeline(
        [
            ("preprocessor", unscaled),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    scale_pos_weight=xgb_weight,
                    eval_metric="aucpr",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }


def compute_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """negatives / positives -- XGBoost's imbalance correction."""
    y = np.asarray(y_train).astype(int)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("Training set contains no positive (default) cases")
    return float((len(y) - positives) / positives)


def fit_models(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Pipeline]:
    """Fit every pipeline on the training fold only."""
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted


def predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive (default) class."""
    return pipeline.predict_proba(X)[:, 1]


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names produced by the fitted preprocessor (used by SHAP)."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(pipeline: Pipeline, path=config.MODEL_FILE) -> Any:
    """Persist the full pipeline (preprocessing + estimator) with joblib.

    Saving the fitted preprocessor together with the estimator guarantees that
    inference applies exactly the transformations used at training time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path=config.MODEL_FILE) -> Pipeline:
    """Load a persisted pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run `python run_pipeline.py` first."
        )
    return joblib.load(path)

"""Central configuration for the Credit Risk & Loan Default project.

Every path, threshold and magic number used by the pipeline lives here so the
project can be reproduced and audited from a single place.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the repository root -- no absolute/OS-specific paths)
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
SQL_DIR: Path = PROJECT_ROOT / "sql"

RAW_DATA_FILE: Path = RAW_DATA_DIR / "credit_risk_dataset.csv"
CLEAN_DATA_FILE: Path = PROCESSED_DATA_DIR / "credit_risk_clean.csv"
FEATURED_DATA_FILE: Path = PROCESSED_DATA_DIR / "credit_risk_featured.csv"
SCORED_DATA_FILE: Path = PROCESSED_DATA_DIR / "credit_risk_scored.csv"

MODEL_FILE: Path = MODELS_DIR / "credit_risk_model.pkl"
FEATURE_CONFIG_FILE: Path = MODELS_DIR / "feature_config.json"
METRICS_FILE: Path = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_FILE: Path = REPORTS_DIR / "model_comparison.csv"
SHAP_IMPORTANCE_FILE: Path = REPORTS_DIR / "shap_global_importance.csv"
SQL_RESULTS_FILE: Path = REPORTS_DIR / "sql_analytics.md"

# --------------------------------------------------------------------------
# Data contract
# --------------------------------------------------------------------------
# Candidate target names -- the loader picks whichever is actually present.
TARGET_CANDIDATES: tuple[str, ...] = (
    "loan_status",
    "loan_default",
    "default",
    "default_flag",
    "is_default",
)

# Columns that identify a customer and must never be used as predictors.
ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "id",
    "customer_id",
    "client_id",
    "loan_id",
    "sk_id_curr",
    "member_id",
)

# Domain validity ranges. Values outside these are treated as data-entry
# errors, set to NaN and imputed inside the modelling pipeline.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "person_age": (18, 100),
    "person_emp_length": (0, 60),
    "person_income": (0, 5_000_000),
    "loan_amnt": (0, 1_000_000),
    "loan_int_rate": (0, 100),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, 60),
}

# --------------------------------------------------------------------------
# Splitting / reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20          # held out for the final unbiased evaluation
VALIDATION_SIZE: float = 0.20    # share of the remaining data used for tuning

# --------------------------------------------------------------------------
# Feature engineering bins (business-defined, NOT learned from the data,
# so they introduce no train/test information leakage)
# --------------------------------------------------------------------------
AGE_BINS: tuple[float, ...] = (17, 25, 35, 45, 60, 120)
AGE_LABELS: tuple[str, ...] = ("18-25", "26-35", "36-45", "46-60", "60+")

INCOME_BINS: tuple[float, ...] = (0, 30_000, 60_000, 100_000, 150_000, float("inf"))
INCOME_LABELS: tuple[str, ...] = ("<30k", "30k-60k", "60k-100k", "100k-150k", "150k+")

LOAN_AMOUNT_BINS: tuple[float, ...] = (0, 5_000, 10_000, 15_000, 25_000, float("inf"))
LOAN_AMOUNT_LABELS: tuple[str, ...] = ("<5k", "5k-10k", "10k-15k", "15k-25k", "25k+")

# --------------------------------------------------------------------------
# Risk segmentation
# --------------------------------------------------------------------------
# PROJECT-DEFINED decision thresholds on the predicted probability of default.
# They are policy choices, not statistical truths: a lender would calibrate
# them against its own loss-given-default and approval-rate targets.
LOW_RISK_THRESHOLD: float = 0.10     # PD < 10%  -> Low Risk
HIGH_RISK_THRESHOLD: float = 0.40    # PD >= 40% -> High Risk
RISK_LABELS: tuple[str, str, str] = ("Low Risk", "Medium Risk", "High Risk")

# Classification cut-off used for the hard 0/1 prediction. The pipeline tunes
# this on the VALIDATION set and overwrites the value stored in
# models/feature_config.json; 0.5 is only the starting default.
DEFAULT_DECISION_THRESHOLD: float = 0.50

# --------------------------------------------------------------------------
# Plot styling
# --------------------------------------------------------------------------
FIGURE_DPI: int = 110
FIGSIZE_WIDE: tuple[int, int] = (11, 5)
FIGSIZE_SQUARE: tuple[int, int] = (6, 5)


def ensure_directories() -> None:
    """Create every output directory the pipeline writes to."""
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

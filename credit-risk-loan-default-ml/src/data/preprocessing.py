"""Data loading, profiling and cleaning.

Design rule followed throughout this module: only operations that are
*row-independent* (dropping duplicates, nulling impossible values, fixing
dtypes) happen here. Anything that learns a statistic from the data
(imputation values, scaling, encoding) is deferred to the scikit-learn
pipeline so that it is fitted on the training fold only -- see
``src/models/train.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def normalise_text_column(series: pd.Series) -> pd.Series:
    """Upper-case and strip a text column, returning object dtype with np.nan.

    scikit-learn's imputers cannot handle pandas' nullable ``string`` dtype
    (``pd.NA`` raises on truth-testing), so text columns are kept as plain
    objects with ``np.nan`` for missing values.
    """
    cleaned = series.astype("object").map(
        lambda v: v.strip().upper() if isinstance(v, str) else np.nan
    )
    return cleaned


def normalise_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`normalise_text_column` to every non-numeric column."""
    out = df.copy()
    for column in out.columns:
        if not pd.api.types.is_numeric_dtype(out[column]):
            out[column] = normalise_text_column(out[column])
    return out


def load_raw_data(path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw loan dataset.

    Args:
        path: CSV file to read.

    Returns:
        The raw dataframe, unmodified.

    Raises:
        FileNotFoundError: if the CSV is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. See README -> Dataset for download steps."
        )
    return pd.read_csv(path)


def detect_target(
    df: pd.DataFrame,
    candidates: tuple[str, ...] = config.TARGET_CANDIDATES,
) -> str:
    """Return the name of the binary target column present in ``df``.

    Raises:
        ValueError: if none of the candidate names is present.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"No target column found. Looked for {candidates}; got {list(df.columns)}"
    )


def detect_id_columns(
    df: pd.DataFrame,
    candidates: tuple[str, ...] = config.ID_COLUMN_CANDIDATES,
) -> list[str]:
    """Return identifier-like columns, which must be excluded from features."""
    lowered = {c.lower(): c for c in df.columns}
    return [lowered[c] for c in candidates if c in lowered]


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Build a machine-readable data-understanding report.

    Returns a dict with shape, dtypes, missing counts, duplicate count,
    cardinality of categoricals and (if given) the target balance.
    """
    profile: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_counts": {
            c: int(n) for c, n in df.isna().sum().items() if n > 0
        },
        "missing_pct": {
            c: round(float(n) / len(df) * 100, 2)
            for c, n in df.isna().sum().items()
            if n > 0
        },
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
    }
    profile["categorical_levels"] = {
        c: sorted(map(str, df[c].dropna().unique()))
        for c in profile["categorical_columns"]
    }
    if target and target in df.columns:
        counts = df[target].value_counts().sort_index()
        profile["target"] = {
            "name": target,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "positive_rate": round(float(df[target].mean()), 4),
            "imbalance_ratio": round(
                float((counts.max() / counts.min())) if counts.min() else float("nan"), 2
            ),
        }
    return profile


def outlier_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """IQR-based outlier counts per numeric column (diagnostic only).

    Outliers are *reported*, not removed: in credit data an extreme income or
    loan amount is usually a genuine, informative observation.
    """
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    rows = []
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < low) | (series > high)
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "lower_fence": low,
                "upper_fence": high,
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(float(mask.mean()) * 100, 2),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_data(
    df: pd.DataFrame,
    target: str,
    valid_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw dataframe without leaking information across rows.

    Steps:
        1. Drop exact duplicate rows.
        2. Drop rows with a missing/invalid target (a label cannot be imputed).
        3. Replace domain-impossible values with NaN (imputed later, in-pipeline).
        4. Drop identifier columns.

    Args:
        df: raw dataframe.
        target: name of the binary target column.
        valid_ranges: mapping column -> (min, max) of plausible values.

    Returns:
        (cleaned dataframe, report dict describing what was changed).
    """
    valid_ranges = valid_ranges or config.VALID_RANGES
    report: dict[str, Any] = {"input_rows": int(len(df))}
    out = df.copy()

    # 1. exact duplicates -------------------------------------------------
    n_dupes = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = n_dupes

    # 2. target validation -------------------------------------------------
    before = len(out)
    out = out[out[target].notna()]
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out[out[target].isin([0, 1])]
    out[target] = out[target].astype(int)
    report["rows_dropped_invalid_target"] = int(before - len(out))

    # 3. impossible values -> NaN -----------------------------------------
    invalid_counts: dict[str, int] = {}
    for col, (low, high) in valid_ranges.items():
        if col not in out.columns:
            continue
        mask = out[col].notna() & ((out[col] < low) | (out[col] > high))
        n_invalid = int(mask.sum())
        if n_invalid:
            out.loc[mask, col] = np.nan
            invalid_counts[col] = n_invalid
    report["invalid_values_nulled"] = invalid_counts

    # 4. identifier columns -------------------------------------------------
    id_cols = detect_id_columns(out)
    if id_cols:
        out = out.drop(columns=id_cols)
    report["id_columns_dropped"] = id_cols

    # 5. tidy categorical text ---------------------------------------------
    out = normalise_text_columns(out)

    out = out.reset_index(drop=True)
    report["output_rows"] = int(len(out))
    report["missing_after_clean"] = {
        c: int(n) for c, n in out.isna().sum().items() if n > 0
    }
    return out, report


def save_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    """Write ``df`` to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

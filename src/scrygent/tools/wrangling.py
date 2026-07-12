"""Deterministic data wrangling engine.

Provides tools for filtering, resetting, and normalizing datasets.
Transforming tools write to temporary CSVs to support the multi-step
composition pattern without passing DataFrames through state.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import NormalizeMethod
from ._shared.filtering import apply_filters
from .io import load_csv, write_temp_csv

logger = logging.getLogger(__name__)


def filter_dataset(
    current_csv_path: Path,
    filters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Filters the dataset and writes the result to a new temporary CSV.

    Args:
        current_csv_path: Path to the active CSV dataset.
        filters: List of filter conditions to apply.

    Returns:
        A dictionary containing the new CSV path, row count, and any warnings.
    """
    if not filters:
        raise ValueError("filter_dataset requires at least one filter condition.")

    logger.info("Executing filter_dataset | filters: %d", len(filters))

    df = load_csv(current_csv_path)
    filtered_df = apply_filters(df, filters)

    warning = None
    if filtered_df.empty:
        warning = "Filtered dataset is empty. Subsequent steps will operate on zero rows."
        logger.warning(warning)

    new_path = write_temp_csv(filtered_df, prefix="scrygent_wrangle_")

    return {
        "current_csv_path": str(new_path),
        "row_count": len(filtered_df),
        "warning": warning,
    }


def reset_dataset(original_csv_path: Path) -> dict[str, Any]:
    """Reverts the active dataset path to the original uploaded CSV.

    Args:
        original_csv_path: The immutable path to the initial user upload.

    Returns:
        A dictionary containing the reset CSV path.
    """
    path = Path(original_csv_path)
    if not path.exists():
        logger.error("reset_dataset failed. original_csv_path missing: %s", path)
        raise FileNotFoundError(
            f"Cannot reset: original_csv_path no longer exists at '{path}'. "
            "This indicates state corruption -- original_csv_path must be immutable."
        )

    logger.info("Executing reset_dataset -> %s", path)
    return {"current_csv_path": str(path)}


def _min_max(series: pd.Series) -> Any:
    """Normalizes numeric data to a 0-1 range."""
    lo, hi = series.min(), series.max()
    if lo == hi:
        raise ValueError(f"Cannot min-max normalize column '{series.name}': all values are identical ({lo}).")
    return (series - lo) / (hi - lo)


def _z_score(series: pd.Series) -> pd.Series:
    """Standardizes numeric data to have a mean of 0 and standard deviation of 1."""
    std = series.std()
    if std == 0 or pd.isna(std):
        raise ValueError(f"Cannot z-score normalize column '{series.name}': zero or undefined variance.")
    return (series - series.mean()) / std


def _log_transform(series: pd.Series) -> pd.Series:
    """Applies a natural log transform to strictly positive numeric data."""
    if (series <= 0).any():
        raise ValueError(
            f"Cannot log-transform column '{series.name}': contains non-positive values. "
            "Log transform requires all values > 0."
        )
    return np.log(series)  # type: ignore


def _strip(series: pd.Series) -> pd.Series:
    """Removes leading and trailing whitespace from string data."""
    return series.astype(str).str.strip()


def _lowercase(series: pd.Series) -> pd.Series:
    """Converts string data to lowercase."""
    return series.astype(str).str.lower()


def _uppercase(series: pd.Series) -> pd.Series:
    """Converts string data to uppercase."""
    return series.astype(str).str.upper()


def _title_case(series: pd.Series) -> pd.Series:
    """Converts string data to title case."""
    return series.astype(str).str.title()


_NUMERIC_METHODS: dict[NormalizeMethod, Callable[[pd.Series], pd.Series]] = {
    NormalizeMethod.MIN_MAX: _min_max,
    NormalizeMethod.Z_SCORE: _z_score,
    NormalizeMethod.LOG: _log_transform,
}

_STRING_METHODS: dict[NormalizeMethod, Callable[[pd.Series], pd.Series]] = {
    NormalizeMethod.STRIP: _strip,
    NormalizeMethod.LOWERCASE: _lowercase,
    NormalizeMethod.UPPERCASE: _uppercase,
    NormalizeMethod.TITLE_CASE: _title_case,
}


def normalize_column(
    current_csv_path: Path,
    column: str,
    method: str,
) -> dict[str, Any]:
    """Applies a transformation method to a specific column and writes to a new CSV.

    Args:
        current_csv_path: Path to the active CSV dataset.
        column: The column to normalize.
        method: The normalization method to apply.

    Returns:
        A dictionary containing the new CSV path, column name, method, and before/after stats.
    """
    try:
        resolved_method = NormalizeMethod(method)
    except ValueError:
        valid = sorted(m.value for m in NormalizeMethod)
        raise ValueError(f"Unsupported normalize method '{method}'. Choose from: {valid}") from None

    logger.info("Executing normalize_column | column: %s | method: %s", column, resolved_method)

    df = load_csv(current_csv_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    series = df[column]
    is_numeric_method = resolved_method in _NUMERIC_METHODS

    if is_numeric_method and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Method '{resolved_method}' requires a numeric column; '{column}' has dtype '{series.dtype}'."
        )
    if not is_numeric_method and pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Method '{resolved_method}' is a string operation; '{column}' has numeric dtype '{series.dtype}'."
        )

    before_stats = None
    if is_numeric_method:
        before_stats = {"min": float(series.min()), "max": float(series.max()), "mean": float(series.mean())}

    transform = (_NUMERIC_METHODS | _STRING_METHODS)[resolved_method]
    df[column] = transform(series)

    after_stats = None
    if is_numeric_method:
        after_stats = {"min": float(df[column].min()), "max": float(df[column].max()), "mean": float(df[column].mean())}

    new_path = write_temp_csv(df, prefix="scrygent_wrangle_")

    return {
        "current_csv_path": str(new_path),
        "column": column,
        "method": resolved_method,
        "before": before_stats,
        "after": after_stats,
    }

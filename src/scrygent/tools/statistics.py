"""Deterministic statistical analysis engine.

Provides pure Python implementations for correlation, regression,
outlier detection, and on-demand column profiling.
"""

import logging
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import CorrelationMethod, OutlierMethod, RegressionMethod
from ._shared.column_stats import compute_detailed_stats
from .io import load_csv

logger = logging.getLogger(__name__)


def _require_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validates that all specified columns exist and are numeric."""
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric (dtype: '{df[col].dtype}').")


def correlation(
    current_csv_path: Path,
    columns: list[str],
    method: str = CorrelationMethod.PEARSON,
) -> dict[str, Any]:
    """Computes the correlation matrix or pairwise correlation for specified columns.

    Args:
        current_csv_path: Path to the active CSV dataset.
        columns: List of column names to correlate.
        method: The correlation algorithm to use (pearson, spearman, kendall).

    Returns:
        A dictionary containing the method, and either a single correlation
        coefficient or a list of pairwise correlations.
    """
    try:
        resolved_method = CorrelationMethod(method)
    except ValueError:
        valid = sorted(m.value for m in CorrelationMethod)
        raise ValueError(f"Unsupported correlation method '{method}'. Choose from: {valid}") from None

    if len(columns) < 2:
        raise ValueError("correlation requires at least 2 columns.")

    logger.info("Executing correlation | columns: %s | method: %s", columns, resolved_method)

    df = load_csv(current_csv_path)
    _require_numeric_columns(df, columns)

    if len(columns) == 2:
        col_a, col_b = columns
        coef = df[col_a].corr(df[col_b], method=resolved_method)  # type: ignore
        return {"method": resolved_method, "column_a": col_a, "column_b": col_b, "correlation": float(coef)}

    corr_matrix = df[columns].corr(method=resolved_method)  # type: ignore
    pairs = [
        {
            "column_a": a,
            "column_b": b,
            "correlation": float(corr_matrix.loc[a, b]),  # type: ignore
        }
        for a, b in combinations(columns, 2)
    ]
    return {"method": resolved_method, "pairs": pairs}


def _fit_linear(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    """Fits a multivariate linear regression using ordinary least squares."""
    x = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])

    coeffs, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    intercept, feature_coeffs = coeffs[0], coeffs[1:]

    y_pred = x_design @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None

    return {
        "intercept": float(intercept),
        "coefficients": {k: float(v) for k, v in zip(features, feature_coeffs)},
        "r_squared": float(r_squared) if r_squared is not None else None,
    }


_REGRESSION_METHODS: dict[RegressionMethod, Callable[[pd.DataFrame, str, list[str]], dict[str, Any]]] = {
    RegressionMethod.LINEAR: _fit_linear,
}


def regression(
    current_csv_path: Path,
    target: str,
    features: list[str],
    method: str = RegressionMethod.LINEAR,
) -> dict[str, Any]:
    """Computes a regression model for a target column based on feature columns.

    Args:
        current_csv_path: Path to the active CSV dataset.
        target: The target column name.
        features: List of feature column names.
        method: The regression algorithm to use.

    Returns:
        A dictionary containing the model coefficients, intercept, and R-squared value.
    """
    try:
        resolved_method = RegressionMethod(method)
    except ValueError:
        valid = sorted(m.value for m in RegressionMethod)
        raise ValueError(f"Unsupported regression method '{method}'. Choose from: {valid}") from None

    if not features:
        raise ValueError("regression requires at least 1 feature column.")
    if target in features:
        raise ValueError(f"Target column '{target}' cannot also appear in features.")

    logger.info("Executing regression | target: %s | features: %s | method: %s", target, features, resolved_method)

    df = load_csv(current_csv_path)
    _require_numeric_columns(df, [target] + features)

    working = df[[target] + features].dropna()
    if len(working) < len(features) + 2:
        raise ValueError(
            f"Insufficient complete rows ({len(working)}) to fit regression with {len(features)} feature(s). "
            "Need at least features + 2 rows after dropping missing values."
        )

    fit_result = _REGRESSION_METHODS[resolved_method](working, target, features)

    return {
        "method": resolved_method,
        "target": target,
        "features": features,
        "row_count": len(working),
        **fit_result,
    }


def _iqr_outliers(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Detects outliers using the Interquartile Range (IQR) method."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, {"lower_bound": float(lower), "upper_bound": float(upper), "q1": float(q1), "q3": float(q3)}


def _zscore_outliers(series: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, dict[str, float]]:
    """Detects outliers using the Z-Score method."""
    std = series.std()
    if std == 0 or pd.isna(std):
        raise ValueError(f"Cannot compute z-score outliers for '{series.name}': zero or undefined variance.")
    z_scores = (series - series.mean()) / std
    mask = z_scores.abs() > threshold
    return mask, {"threshold": threshold, "mean": float(series.mean()), "std": float(std)}


_OUTLIER_METHODS: dict[OutlierMethod, Callable[..., tuple[pd.Series, dict[str, float]]]] = {
    OutlierMethod.IQR: _iqr_outliers,
    OutlierMethod.Z_SCORE: _zscore_outliers,
}

MAX_OUTLIER_EXAMPLES = 20


def detect_outliers(
    current_csv_path: Path,
    column: str,
    method: str = OutlierMethod.IQR,
) -> dict[str, Any]:
    """Detects statistical outliers in a specified numeric column.

    Args:
        current_csv_path: Path to the active CSV dataset.
        column: The column to analyze.
        method: The outlier detection algorithm to use (iqr, z_score).

    Returns:
        A dictionary containing the outlier count, examples, and method parameters.
    """
    try:
        resolved_method = OutlierMethod(method)
    except ValueError:
        valid = sorted(m.value for m in OutlierMethod)
        raise ValueError(f"Unsupported outlier method '{method}'. Choose from: {valid}") from None

    logger.info("Executing detect_outliers | column: %s | method: %s", column, resolved_method)

    df = load_csv(current_csv_path)
    _require_numeric_columns(df, [column])

    series = df[column].dropna()
    mask, params = _OUTLIER_METHODS[resolved_method](series)
    outliers = series[mask]

    return {
        "column": column,
        "method": resolved_method,
        "outlier_count": int(mask.sum()),
        "outlier_examples": [float(x) for x in outliers.head(MAX_OUTLIER_EXAMPLES).tolist()],
        "params": params,
    }


def request_column_stats(current_csv_path: Path, columns: list[str]) -> dict[str, Any]:
    """Lazily fetches detailed statistical metrics for specified columns.

    Args:
        current_csv_path: Path to the active CSV dataset.
        columns: List of column names to profile.

    Returns:
        A dictionary containing the detailed statistics for the requested columns.
    """
    if not columns:
        raise ValueError("request_column_stats requires at least 1 column.")

    logger.info("Executing request_column_stats | columns: %s", columns)

    df = load_csv(current_csv_path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}. Available: {list(df.columns)}")

    df_norm = df.rename(columns=str)
    stats = compute_detailed_stats(df_norm, columns)

    return {"detailed_stats": stats}

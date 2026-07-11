import logging
import numpy as np
import pandas as pd

from itertools import combinations
from typing import Any, Callable
from pathlib import Path

from .io import load_csv
from ._shared.column_stats import compute_detailed_stats
from ..contracts.statistics import CorrelationMethod, RegressionMethod, OutlierMethod

logger = logging.getLogger(__name__)


def _require_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric (dtype: '{df[col].dtype}').")


# --- correlation ---

SUPPORTED_CORRELATION_METHODS = set(CorrelationMethod)


def correlation(
    current_csv_path: Path,
    columns: list[str],
    method: str = CorrelationMethod.PEARSON,
) -> dict[str, Any]:
    raw_method = method
    try:
        method = CorrelationMethod(method)
    except ValueError:
        raise ValueError(
            f"Unsupported correlation method '{raw_method}'. "
            f"Choose from: {sorted(m.value for m in CorrelationMethod)}"
        ) from None

    if len(columns) < 2:
        raise ValueError("correlation requires at least 2 columns.")

    logger.info("Executing correlation | columns: %s | method: %s", columns, method)

    df = load_csv(current_csv_path)
    _require_numeric_columns(df, columns)

    if len(columns) == 2:
        col_a, col_b = columns
        coef = df[col_a].corr(df[col_b], method=method)  # type: ignore
        return {"method": method, "column_a": col_a, "column_b": col_b, "correlation": coef}

    corr_matrix = df[columns].corr(method=method)  # type: ignore
    pairs = [
        {"column_a": a, "column_b": b, "correlation": corr_matrix.loc[a, b]}
        for a, b in combinations(columns, 2)
    ]
    return {"method": method, "pairs": pairs}


# --- regression ---

def _fit_linear(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])

    coeffs, residuals, _rank, _sv = np.linalg.lstsq(X_design, y, rcond=None)
    intercept, feature_coeffs = coeffs[0], coeffs[1:]

    y_pred = X_design @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None

    return {
        "intercept": intercept,
        "coefficients": dict(zip(features, feature_coeffs)),
        "r_squared": r_squared,
    }


_REGRESSION_METHODS: dict[RegressionMethod, Callable[[pd.DataFrame, str, list[str]], dict[str, Any]]] = {
    RegressionMethod.LINEAR: _fit_linear,
}

SUPPORTED_REGRESSION_METHODS = set(_REGRESSION_METHODS)


def regression(
    current_csv_path: Path,
    target: str,
    features: list[str],
    method: str = RegressionMethod.LINEAR,
) -> dict[str, Any]:
    raw_method = method
    try:
        method = RegressionMethod(method)
    except ValueError:
        raise ValueError(
            f"Unsupported regression method '{raw_method}'. "
            f"Choose from: {sorted(m.value for m in RegressionMethod)}"
        ) from None

    if not features:
        raise ValueError("regression requires at least 1 feature column.")
    if target in features:
        raise ValueError(f"Target column '{target}' cannot also appear in features.")

    logger.info("Executing regression | target: %s | features: %s | method: %s", target, features, method)

    df = load_csv(current_csv_path)
    _require_numeric_columns(df, [target] + features)

    working = df[[target] + features].dropna()
    if len(working) < len(features) + 2:
        raise ValueError(
            f"Insufficient complete rows ({len(working)}) to fit regression with {len(features)} feature(s). "
            "Need at least features + 2 rows after dropping missing values."
        )

    fit_result = _REGRESSION_METHODS[method](working, target, features)

    return {
        "method": method,
        "target": target,
        "features": features,
        "row_count": len(working),
        **fit_result,
    }


# --- outlier detection ---

def _iqr_outliers(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, {"lower_bound": lower, "upper_bound": upper, "q1": q1, "q3": q3}


def _zscore_outliers(series: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, dict[str, float]]:
    std = series.std()
    if std == 0 or pd.isna(std):
        raise ValueError(f"Cannot compute z-score outliers for '{series.name}': zero or undefined variance.")
    z_scores = (series - series.mean()) / std
    mask = z_scores.abs() > threshold
    return mask, {"threshold": threshold, "mean": series.mean(), "std": std}


_OUTLIER_METHODS: dict[OutlierMethod, Callable[..., tuple[pd.Series, dict[str, float]]]] = {
    OutlierMethod.IQR: _iqr_outliers,
    OutlierMethod.Z_SCORE: _zscore_outliers,
}

SUPPORTED_OUTLIER_METHODS = set(_OUTLIER_METHODS)

MAX_OUTLIER_EXAMPLES = 20


def detect_outliers(
    current_csv_path: Path,
    column: str,
    method: str = OutlierMethod.IQR,
) -> dict[str, Any]:
    raw_method = method
    try:
        method = OutlierMethod(method)
    except ValueError:
        raise ValueError(
            f"Unsupported outlier method '{raw_method}'. "
            f"Choose from: {sorted(m.value for m in OutlierMethod)}"
        ) from None

    logger.info("Executing detect_outliers | column: %s | method: %s", column, method)

    df = load_csv(current_csv_path)
    _require_numeric_columns(df, [column])

    series = df[column].dropna()
    mask, params = _OUTLIER_METHODS[method](series)
    outliers = series[mask]

    return {
        "column": column,
        "method": method,
        "outlier_count": int(mask.sum()),
        "outlier_examples": outliers.head(MAX_OUTLIER_EXAMPLES).tolist(),
        "params": params,
    }


# --- request_column_stats (lazy fetch) ---

def request_column_stats(current_csv_path: Path, columns: list[str]) -> dict[str, Any]:
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

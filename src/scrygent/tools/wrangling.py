import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._shared.filtering import apply_filters
from .io import load_csv, write_temp_csv
from ..contracts.wrangling import NormalizeMethod

logger = logging.getLogger(__name__)


# --- filter_dataset ---

def filter_dataset(
    current_csv_path: Path,
    filters: list[dict[str, Any]],
) -> dict[str, Any]:
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


# --- reset_dataset ---

def reset_dataset(original_csv_path: Path) -> dict[str, Any]:
    path = Path(original_csv_path)
    if not path.exists():
        logger.error("reset_dataset failed. original_csv_path missing: %s", path)
        raise FileNotFoundError(
            f"Cannot reset: original_csv_path no longer exists at '{path}'. "
            "This indicates state corruption -- original_csv_path must be immutable."
        )

    logger.info("Executing reset_dataset -> %s", path)
    return {"current_csv_path": str(path)}


# --- normalize_column ---

def _min_max(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if lo == hi:
        raise ValueError(
            f"Cannot min-max normalize column '{series.name}': all values are identical ({lo})."
        )
    return (series - lo) / (hi - lo)


def _z_score(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        raise ValueError(
            f"Cannot z-score normalize column '{series.name}': zero or undefined variance."
        )
    return (series - series.mean()) / std


def _log_transform(series: pd.Series) -> pd.Series:
    if (series <= 0).any():
        raise ValueError(
            f"Cannot log-transform column '{series.name}': contains non-positive values. "
            "Log transform requires all values > 0."
        )
    return np.log(series) # type: ignore


def _strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _lowercase(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower()


def _uppercase(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper()


def _title_case(series: pd.Series) -> pd.Series:
    return series.astype(str).str.title()


# Dispatch dicts now keyed by NormalizeMethod enum members, not bare
# strings. NormalizeMethod is a StrEnum so these keys still equal and
# hash the same as the plain strings a JSON payload deserializes into --
# no behavior change, just a single source of truth for valid values,
# shared with schemas.NormalizeColumnParams.method via contracts.wrangling.
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

SUPPORTED_NORMALIZE_METHODS = set(_NUMERIC_METHODS) | set(_STRING_METHODS)


def normalize_column(
    current_csv_path: Path,
    column: str,
    method: str,
) -> dict[str, Any]:
    raw_method = method
    try:
        method = NormalizeMethod(method)
    except ValueError:
        raise ValueError(
            f"Unsupported normalize method '{raw_method}'. "
            f"Choose from: {sorted(m.value for m in NormalizeMethod)}"
        ) from None

    logger.info("Executing normalize_column | column: %s | method: %s", column, method)

    df = load_csv(current_csv_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    series = df[column]
    is_numeric_method = method in _NUMERIC_METHODS

    if is_numeric_method and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Method '{method}' requires a numeric column; '{column}' has dtype '{series.dtype}'."
        )
    if not is_numeric_method and pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Method '{method}' is a string operation; '{column}' has numeric dtype '{series.dtype}'."
        )

    before_stats = None
    if is_numeric_method:
        before_stats = {"min": series.min(), "max": series.max(), "mean": series.mean()}

    transform = (_NUMERIC_METHODS | _STRING_METHODS)[method]
    df[column] = transform(series)

    after_stats = None
    if is_numeric_method:
        after_stats = {"min": df[column].min(), "max": df[column].max(), "mean": df[column].mean()}

    new_path = write_temp_csv(df, prefix="scrygent_wrangle_")

    return {
        "current_csv_path": str(new_path),
        "column": column,
        "method": method,
        "before": before_stats,
        "after": after_stats,
    }

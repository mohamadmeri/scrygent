"""Shared per-column statistics engine.

Used by the Profiler Node for initial profiling and by the
request_column_stats tool for lazy fetching. Guarantees identical
statistical shapes regardless of the profiling path.
"""

from typing import Any

import pandas as pd


def _to_python_scalar(value: Any) -> Any:
    """Converts NumPy and Pandas scalar types to native Python primitives."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalize_number(value: Any) -> Any:
    """Converts numeric values to native Python types and rounds floats to 4 decimals."""
    value = _to_python_scalar(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def compute_detailed_stats(df: pd.DataFrame, target_columns: list[str]) -> dict[str, dict[str, Any]]:
    """Computes detailed statistical metrics for a specified list of columns.

    Args:
        df: The source DataFrame.
        target_columns: The exact column names to profile.

    Returns:
        A dictionary mapping column names to their statistical summaries.
    """
    stats: dict[str, dict[str, Any]] = {}
    total_rows = len(df)

    for col in target_columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        unique_count = _to_python_scalar(col_data.nunique())

        col_stats: dict[str, Any] = {
            "dtype": str(col_data.dtype),
            "null_rate": round(null_count / total_rows, 4) if total_rows > 0 else 0.0,
            "unique_count": unique_count,
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_stats["min"] = _normalize_number(col_data.min())
            col_stats["max"] = _normalize_number(col_data.max())
            col_stats["mean"] = _normalize_number(col_data.mean())

        stats[str(col)] = col_stats

    return stats

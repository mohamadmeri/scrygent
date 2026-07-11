"""Shared per-column statistics engine. Used by profiler.py (initial
profiling of priority columns) and statistics.py (request_column_stats
lazy fetch). Ensures a lazily-fetched column has an identical stat shape
to one profiled up front -- the Planner can't tell the two paths apart."""

from typing import Any
import pandas as pd


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalar types into native Python scalars for JSON serialization."""
    return value.item() if hasattr(value, "item") else value


def _normalize_number(value: Any) -> Any:
    """Convert numeric values to native Python types and round floats to 4 decimal places."""
    value = _to_python_scalar(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


def compute_detailed_stats(df: pd.DataFrame, target_columns: list[str]) -> dict[str, dict[str, Any]]:
    stats = {}
    total_rows = len(df)

    for col in target_columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        unique_count = _to_python_scalar(col_data.nunique())
        
        col_stats = {
            "dtype": str(col_data.dtype),
            "null_rate": round(null_count / total_rows, 4) if total_rows > 0 else 0.0,
            "unique_count": unique_count
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_stats["min"] =  _normalize_number(col_data.min())
            col_stats["max"] =  _normalize_number(col_data.max())
            col_stats["mean"] = _normalize_number(col_data.mean())

        stats[str(col)] = col_stats

    return stats

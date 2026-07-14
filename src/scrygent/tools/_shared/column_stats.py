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

    Includes structural metadata to prevent LLM hallucinations and guide the Optimizer.

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

            # Insight 1: Sequential ID Detection (Prevents bad math on IDs)
            if total_rows > 0 and col_data.is_monotonic_increasing and unique_count == total_rows:
                col_stats["is_sequential_id"] = True
        else:
            # Tier 1 Defense: Ground-truth vocabulary for low-cardinality columns
            top_values = col_data.dropna().value_counts().head(5).index.tolist()
            col_stats["sample_values"] = [_to_python_scalar(v) for v in top_values]

            # Insight 2: Constant Column Detection
            if unique_count == 1 and total_rows > 0:
                col_stats["is_constant"] = True
                col_stats["constant_value"] = _to_python_scalar(col_data.iloc[0])

            # Insight 3: High-Skew / Imbalance Flag
            if unique_count <= 10 and total_rows > 0:
                top_val_count = col_data.value_counts().iloc[0]
                if (top_val_count / total_rows) > 0.90:
                    col_stats["highly_imbalanced"] = True
                    col_stats["dominant_value"] = _to_python_scalar(col_data.value_counts().index[0])

        stats[str(col)] = col_stats

    return stats

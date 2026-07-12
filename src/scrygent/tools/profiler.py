"""Deterministic dataset profiling engine.

Generates a two-level structural profile (global schema and detailed
statistics) to minimize prompt size while ensuring the Planner never
guesses data distributions.
"""

import logging
import re
from typing import Any

import pandas as pd

from ._shared.column_stats import compute_detailed_stats
from .io import get_column_sample

logger = logging.getLogger(__name__)

MAX_DETAILED_COLUMNS = 15
MIN_ROWS_FOR_STATISTICAL_ID_SIGNAL = 20

_ID_PATTERNS = ("id", "_id", "uuid", "guid", "hash", "key")


def _is_identifier(name: str, series: pd.Series) -> bool:
    """Heuristic check to determine if a column is likely an identifier."""
    name_l = name.lower()

    if any(p == name_l or name_l.endswith(p) for p in _ID_PATTERNS):
        return True

    n = len(series)
    if n < MIN_ROWS_FOR_STATISTICAL_ID_SIGNAL:
        return False

    unique_ratio = series.nunique(dropna=False) / n

    if unique_ratio > 0.97:
        return True

    if pd.api.types.is_integer_dtype(series):
        if series.is_monotonic_increasing and unique_ratio > 0.95:
            return True

    return False


def _query_score(col: str, series: pd.Series, query: str) -> float:
    """Calculates a relevance score for a column based on the user query."""
    q = query.lower()
    c = col.lower()
    score = 0.0

    pattern = rf"\b{re.escape(c)}\b"
    if re.search(pattern, q):
        score += 3.0
    elif any(tok in q for tok in c.split("_") if len(tok) > 2):
        score += 1.0

    if pd.api.types.is_numeric_dtype(series):
        if any(x in q for x in ["greater", "less", "above", "below", "rate", "$"]):
            score += 1.5

    return score


def _get_global_schema(df: pd.DataFrame) -> dict[str, str]:
    """Extracts the column name to dtype mapping for the entire dataset."""
    return {str(c): str(t) for c, t in df.dtypes.items()}


def _select_columns(df: pd.DataFrame, query: str, max_cols: int) -> list[str]:
    """Selects the most relevant columns for detailed statistical profiling."""
    scored = []

    for col in df.columns:
        s = df[col]
        score = _query_score(col, s, query)

        if _is_identifier(col, s):
            score -= 5.0

        score += s.notna().mean() * 0.5

        if pd.api.types.is_numeric_dtype(s):
            score += 0.3

        scored.append((col, score))

    scored.sort(key=lambda x: (-x[1], list(df.columns).index(x[0])))

    return [c for c, _ in scored[:max_cols]]


def profile_dataframe(df: pd.DataFrame, user_query: str) -> dict[str, Any]:
    """Generates a comprehensive, two-level structural profile of a DataFrame.

    Args:
        df: The source DataFrame to profile.
        user_query: The natural language query used to prioritize columns.

    Returns:
        A dictionary containing the row count, global schema, detailed stats,
        truncation flag, row sample, and missing detailed stats list.
    """
    logger.info("Profiling df: rows=%d cols=%d", len(df), len(df.columns))

    if df.empty:
        return {
            "row_count": 0,
            "global_schema": {},
            "detailed_stats": {},
            "truncated": False,
            "row_sample": [],
            "missing_detailed_stats": [],
        }

    df = df.rename(columns=str)
    global_schema = _get_global_schema(df)
    priority_cols = _select_columns(df, user_query, MAX_DETAILED_COLUMNS)
    detailed_stats = compute_detailed_stats(df, priority_cols)
    row_sample = get_column_sample(df, n=3)
    truncated = len(priority_cols) < len(df.columns)
    missing = sorted(set(global_schema) - set(detailed_stats))

    return {
        "row_count": len(df),
        "global_schema": global_schema,
        "detailed_stats": detailed_stats,
        "truncated": truncated,
        "row_sample": row_sample,
        "missing_detailed_stats": missing,
    }

import logging
import re
from typing import Any

import pandas as pd

from ._shared.column_stats import compute_detailed_stats
from .io import get_column_sample

logger = logging.getLogger(__name__)

MAX_DETAILED_COLUMNS = 15

# Minimum row count before uniqueness/monotonicity are treated as identifier
# evidence. Below this, "every value differs" is just what small samples look
# like -- it isn't a meaningful statistical signal, so we ignore it entirely
# and defer to naming conventions only.
MIN_ROWS_FOR_STATISTICAL_ID_SIGNAL = 20

# IDENTIFIER HEURISTICS
_ID_PATTERNS = ("id", "_id", "uuid", "guid", "hash", "key")


def _is_identifier(name: str, series: pd.Series) -> bool:
    name_l = name.lower()

    # Name-based signal: reliable at any sample size.
    if any(p == name_l or name_l.endswith(p) for p in _ID_PATTERNS):
        return True

    n = len(series)
    if n == 0:
        return False

    # Statistical signal: only meaningful once the sample is large enough
    # that "mostly unique values" isn't just what any small dataset looks
    # like. On a 4-row fixture, a clean numeric column with no repeats is
    # not evidence of an identifier -- it's just noise.
    if n < MIN_ROWS_FOR_STATISTICAL_ID_SIGNAL:
        return False

    unique_ratio = series.nunique(dropna=False) / n

    # strong signal
    if unique_ratio > 0.97:
        return True

    # numeric monotonic index-like
    if pd.api.types.is_integer_dtype(series):
        if series.is_monotonic_increasing and unique_ratio > 0.95:
            return True

    return False


# QUERY SIGNALING
def _query_score(col: str, series: pd.Series, query: str) -> float:
    q = query.lower()
    c = col.lower()

    score = 0.0

    # 1. Strict whole-word match (prevents 'id' matching 'dividend')
    # We use re.escape to handle columns that might have special characters
    pattern = rf"\b{re.escape(c)}\b"
    if re.search(pattern, q):
        score += 3.0

    # 2. Fallback: fuzzy containment for underscores (e.g., 'total_sales' in query 'total sales')
    elif any(tok in q for tok in c.split("_") if len(tok) > 2):
        score += 1.0

    # numeric intent boost
    if pd.api.types.is_numeric_dtype(series):
        if any(x in q for x in ["greater", "less", "above", "below", "rate", "$"]):
            score += 1.5

    return score


# SCHEMA
def _get_global_schema(df: pd.DataFrame) -> dict[str, str]:
    return {str(c): str(t) for c, t in df.dtypes.items()}


# COLUMN SELECTION
def _select_columns(df: pd.DataFrame, query: str, max_cols: int) -> list[str]:
    scored = []

    for col in df.columns:
        s = df[col]
        score = _query_score(col, s, query)

        if _is_identifier(col, s):
            score -= 5.0  # push to bottom but don't remove

        # signal density
        score += s.notna().mean() * 0.5

        # numeric usefulness
        if pd.api.types.is_numeric_dtype(s):
            score += 0.3

        scored.append((col, score))

    # stable ordering: score DESC then original order
    scored.sort(key=lambda x: (-x[1], list(df.columns).index(x[0])))

    return [c for c, _ in scored[:max_cols]]


def profile_dataframe(df: pd.DataFrame, user_query: str) -> dict[str, Any]:
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

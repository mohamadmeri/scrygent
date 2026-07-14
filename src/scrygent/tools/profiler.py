"""Deterministic dataset profiling engine.

Generates a two-level structural profile (global schema and detailed
statistics) to minimize prompt size while ensuring the Planner never
guesses data distributions.
"""

import logging
import re
from collections import Counter
from typing import Any

import pandas as pd

from ._shared.column_stats import _to_python_scalar, compute_detailed_stats
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


def _extract_regex_skeleton(series: pd.Series) -> str | None:
    """Extracts the dominant structural pattern of a string column."""
    sample = series.dropna().head(100).astype(str)
    if sample.empty:
        return None

    skeletons = []
    for val in sample:
        skel = re.sub(r"\d", "#", val)
        skel = re.sub(r"[a-zA-Z]", "A", skel)
        skeletons.append(skel)

    most_common = Counter(skeletons).most_common(1)
    if most_common:
        pattern, count = most_common[0]
        if count / len(sample) >= 0.5:
            return pattern
    return None


def _extract_query_specific_matches(
    df: pd.DataFrame, priority_cols: list[str], user_query: str
) -> dict[str, list[Any]]:
    """Scans priority categorical columns for exact/case-insensitive matches against the user's query tokens.

    Extracts exact ground-truth strings for high-cardinality columns without bloating the prompt.
    """
    raw_tokens = set(re.findall(r"\b\w+\b", user_query.lower()))
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "show",
        "me",
        "find",
        "filter",
        "by",
        "where",
        "what",
        "in",
        "for",
        "and",
        "or",
        "to",
        "of",
        "with",
        "from",
        "that",
        "this",
        "which",
        "who",
        "whose",
        "whom",
        "highest",
        "lowest",
        "top",
    }
    target_tokens = raw_tokens - stop_words

    if not target_tokens:
        return {}

    matches: dict[str, list[Any]] = {}
    for col in priority_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_matches = []
        for val in df[col].dropna().unique():
            val_str = str(val).lower()
            if any(tok in val_str or val_str in tok for tok in target_tokens):
                col_matches.append(_to_python_scalar(val))
                if len(col_matches) >= 5:
                    break

        if col_matches:
            matches[col] = col_matches

    return matches


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
            "query_specific_matches": {},
            "regex_skeletons": {},
        }

    df = df.rename(columns=str)
    global_schema = _get_global_schema(df)
    priority_cols = _select_columns(df, user_query, MAX_DETAILED_COLUMNS)

    detailed_stats = compute_detailed_stats(df, priority_cols)

    # Insight 2: Extract Regex Skeletons for all priority string columns
    regex_skeletons: dict[str, str] = {}
    for col in priority_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            skeleton = _extract_regex_skeleton(df[col])
            if skeleton:
                regex_skeletons[col] = skeleton

    # Tier 1 Defense: Extract exact matches for high-cardinality categorical columns
    query_specific_matches = _extract_query_specific_matches(df, priority_cols, user_query)

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
        "query_specific_matches": query_specific_matches,
        "regex_skeletons": regex_skeletons,
    }

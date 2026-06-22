import logging
import re
from typing import Any
import pandas as pd
import numpy as np

from .io import get_column_sample

logger = logging.getLogger(__name__)

# Configurable threshold for token safety (Tier 1 Architecture Rule)
MAX_DETAILED_COLUMNS = 15


def _safe_cast_metric(value: Any) -> float | None:
    """
    Safely casts Pandas/Numpy numerical metrics to standard Python floats.
    Handles NaN, NaT, None, and Infinity, converting invalid JSON numbers to None.
    """
    if pd.isna(value):
        return None
    try:
        if isinstance(value, np.integer):
            return int(value)
        float_val = float(value)
        if np.isinf(float_val) or np.isnan(float_val):
            return None
        return float_val
    except (ValueError, TypeError):
        return None

def _get_global_schema(df: pd.DataFrame) -> dict[str, str]:
    """
    Extracts the name and dtype of EVERY column in the DataFrame.
    Ensures the Planner has a complete map of the data, preventing the 
    'Profiler Blind Spot'.
    """
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


# We use lexical search instead of semantic similarity search. The reasoning is that instead of using a dedicated LLM call to 
# semantically parse the query for column references, we assume that the planner with the given metadata can infer column 
# references just like it would if we call another LLM. This way, we save on an LLM call and also avoid the risk of the 
# semantic search misinterpreting the query and missing critical column references. Either way in a worst-case scenario
# both LLM calls could fail to identify the correct columns, so we might as well save on the LLM call and rely on 
# the planner's reasoning capabilities.
def _extract_query_columns(df_columns: list[str], user_query: str) -> list[str]:
    """
    Identifies which columns are explicitly referenced in the 
    user's natural language query via strict word-isolation regex matching.
    """
    query_lower = user_query.lower()
    matched = []
    
    for col in df_columns:
        # (?<!\w) and (?!\w) ensure we match the exact column name 
        # without failing on columns that start/end with symbols like () or $.
        pattern = re.compile(r'(?<!\w)' + re.escape(str(col).lower()) + r'(?!\w)')
        if pattern.search(query_lower):
            matched.append(str(col))
            
    return matched


def _select_priority_columns(df: pd.DataFrame, query_cols: list[str], max_cols: int) -> list[str]:
    """
    Determines which columns get full statistical profiles.
    Merges the user-queried columns with the top N most populated columns 
    to stay under token limits.
    """
    # Deduplicate while preserving order
    priority = list(dict.fromkeys(query_cols))
    
    if len(priority) >= max_cols:
        logger.warning(
            "Query references %d columns but MAX_DETAILED_COLUMNS is %d. "
            "Dropping %d query columns from detailed profile.",
            len(priority), max_cols, len(priority) - max_cols
        )
        return priority[:max_cols]
        
    remaining_slots = max_cols - len(priority)
    other_cols = [str(c) for c in df.columns if str(c) not in priority]
    
    if not other_cols:
        return priority
        
    # Heuristic: prioritize remaining columns by the amount of non-null data they contain
    other_cols_sorted = df[other_cols].count().sort_values(ascending=False).index.tolist()
    
    priority.extend(str(c) for c in other_cols_sorted[:remaining_slots])
    return priority


def _compute_detailed_stats(df: pd.DataFrame, target_columns: list[str]) -> dict[str, dict[str, Any]]:
    """
    Calculates null rates, unique counts, and bounds (min/max for numeric) 
    strictly for the provided target columns.
    """
    stats = {}
    total_rows = len(df)
    
    for col in target_columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        
        col_stats = {
            "dtype": str(col_data.dtype),
            "null_rate": round(null_count / total_rows, 4) if total_rows > 0 else 0.0,
            "unique_count": int(col_data.nunique())
        }
        
        # Add bounded stats only for numeric types safely cast to standard Python floats
        if pd.api.types.is_numeric_dtype(col_data):
            col_stats["min"] = _safe_cast_metric(col_data.min())
            col_stats["max"] = _safe_cast_metric(col_data.max())
            col_stats["mean"] = _safe_cast_metric(col_data.mean())
            
        stats[str(col)] = col_stats
        
    return stats


def profile_dataframe(df: pd.DataFrame, user_query: str) -> dict[str, Any]:
    """
    The public orchestrator for the Profiler Node.
    
    Executes the deterministic profiling pipeline:
    1. Extracts global schema.
    2. Identifies priority columns based on the query and token thresholds.
    3. Computes detailed stats for priority columns.
    4. Extracts a 3-row data sample for LLM formatting context.
    
    Returns a structured dictionary mapping to the CSVProfile Pydantic model.
    """
    logger.info("Initiating dataframe profiling. Total columns: %d, Total rows: %d", len(df.columns), len(df))
    
    if df.empty:
        logger.warning("Dataframe is empty. Returning blank profile.")
        return {
            "global_schema": {},
            "detailed_stats": {},
            "truncated": False,
            "row_sample": []
        }

    # Normalize column names to strings to prevent indexing KeyErrors with integer columns
    df_norm = df.rename(columns=str)
    
    # 1. Global Map
    global_schema = _get_global_schema(df_norm)
    
    # 2. Routing & Truncation
    df_columns_str = list(df_norm.columns)
    query_cols = _extract_query_columns(df_columns_str, user_query)
    priority_cols = _select_priority_columns(df_norm, query_cols, MAX_DETAILED_COLUMNS)
    
    # 3. Execution
    detailed_stats = _compute_detailed_stats(df_norm, priority_cols)
    
    # 4. Context Sample 
    row_sample = get_column_sample(df_norm, n=3)
    
    # 5. Assembly
    truncated = len(priority_cols) < len(df_norm.columns)
    if truncated:
        logger.info("Profile truncated. Detailed stats provided for %d out of %d columns.", len(priority_cols), len(df_norm.columns))
    
    return {
        "global_schema": global_schema,
        "detailed_stats": detailed_stats,
        "truncated": truncated,
        "row_sample": row_sample
    }

import logging
from typing import Any, Literal
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_OPERATIONS = {
    "mean", "sum", "count", "nunique", "median",
    "min", "max", "std", "var"
}

def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a sequence of declarative filters.
    Each filter must have keys: column, operator, value.
    Supported operators: ==, !=, >, <, >=, <=, in, contains.
    Handles null filtering: value=None with == or != works as isna/notna.
    """
    working_df = df.copy()
    for f in filters:
        # Validate required keys exist (separate from value extraction to allow None)
        if not all(k in f for k in ("column", "operator", "value")):
            raise ValueError(f"Invalid filter specification (missing keys): {f}")

        col = f["column"]
        op = f["operator"]
        val = f["value"]

        if col not in working_df.columns:
            raise ValueError(f"Filter column '{col}' not found in dataset.")

        # Handle null values for equality/inequality operators
        if val is None:
            if op == "==":
                working_df = working_df[working_df[col].isna()]
            elif op == "!=":
                working_df = working_df[working_df[col].notna()]
            else:
                raise ValueError(
                    f"Operator '{op}' with None value is not supported. "
                    f"Use '==' or '!=' for null checks."
                )
            continue

        # Standard operator dispatch
        if op == "==":
            working_df = working_df[working_df[col] == val]
        elif op == "!=":
            working_df = working_df[working_df[col] != val]
        elif op == ">":
            working_df = working_df[working_df[col] > val]
        elif op == "<":
            working_df = working_df[working_df[col] < val]
        elif op == ">=":
            working_df = working_df[working_df[col] >= val]
        elif op == "<=":
            working_df = working_df[working_df[col] <= val]
        elif op == "in":
            if not isinstance(val, list):
                raise ValueError(f"Operator 'in' requires a list of values, got {type(val)}.")
            working_df = working_df[working_df[col].isin(val)]
        elif op == "contains":
            working_df = working_df[working_df[col].astype(str).str.contains(
                str(val), case=False, na=False
            )]
        else:
            raise ValueError(f"Unsupported filter operator: '{op}'")

    return working_df


def _perform_aggregation(
    df: pd.DataFrame,
    target_column: str,
    operation: str,
    group_by: list[str] | None
) -> Any:
    """
    Execute aggregation. If group_by provided, return a pd.Series indexed by groups.
    Otherwise return a scalar (float/int).
    """
    if group_by:
        grouped = df.groupby(group_by, dropna=False)[target_column]
        if operation == "nunique":
            return grouped.nunique()
        return grouped.agg(operation)
    else:
        series = df[target_column]
        if operation == "nunique":
            return series.nunique()
        return series.agg(operation)


def _format_and_sort_results(
    raw_result: Any,
    sort_order: Literal["asc", "desc"] | None,
    top_k: int | None
) -> Any:
    """
    Apply sorting and truncation to a Series result, then convert to dict.
    Scalars are returned unchanged. All keys are cast to strings for JSON safety.
    """
    if not isinstance(raw_result, pd.Series):
        return raw_result  # Scalar

    # Sort
    if sort_order:
        ascending = sort_order == "asc"
        raw_result = raw_result.sort_values(ascending=ascending)

    # Truncate
    if top_k is not None:
        raw_result = raw_result.head(top_k)

    # Convert to dict, forcing all keys to strings
    return {str(k): v for k, v in raw_result.to_dict().items()}


def analyze_data(
    df: pd.DataFrame,
    target_column: str,
    operation: str,
    filters: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    sort_order: Literal["asc", "desc"] | None = None,
    top_k: int | None = None
) -> dict[str, Any]:
    """
    Unified Declarative Analyst tool for Tier 1.
    Executes filter -> group -> aggregate -> sort -> limit in a single pass.
    Returns a dictionary: {"result": <scalar or dict>}.
    Any remaining NumPy types are sanitized by ScrygentBaseModel on state entry.
    """
    logger.info(
        "analyze_data: target=%s, op=%s, filters=%s, group_by=%s, sort=%s, top_k=%s",
        target_column, operation, filters is not None, group_by, sort_order, top_k
    )

    # 1. Validate inputs
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported operation '{operation}'. Choose from: {SUPPORTED_OPERATIONS}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    if group_by:
        for col in group_by:
            if col not in df.columns:
                raise ValueError(f"Group-by column '{col}' not found in dataset.")
    if filters is None:
        filters = []

    # 2. Apply filters
    working_df = _apply_filters(df, filters) if filters else df.copy()

    if working_df.empty:
        return {"result": None, "warning": "Filtered dataset is empty."}

    # 3. Aggregate
    raw_result = _perform_aggregation(working_df, target_column, operation, group_by)

    # 4. Format & sort
    final_result = _format_and_sort_results(raw_result, sort_order, top_k)

    return {"result": final_result}

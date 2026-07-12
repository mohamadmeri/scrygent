"""Deterministic Pandas compiler for the Analytical Query IR.

Executes the strict pipeline: Filter -> Group -> Aggregate -> Sort -> Limit.
"""

import logging
from typing import Any

import pandas as pd

from ..contracts import Aggregation
from ._shared.filtering import apply_filters

logger = logging.getLogger(__name__)

SUPPORTED_OPERATIONS = set(Aggregation)


def _perform_aggregation(
    df: pd.DataFrame, metrics: list[dict[str, Any]], group_by: list[str] | None
) -> pd.DataFrame | dict[str, Any]:
    """Executes the aggregation phase of the analytical query."""
    if group_by:
        agg_kwargs = {m["alias"]: (m["column"], m["aggregation"]) for m in metrics}
        grouped = df.groupby(group_by, dropna=False)
        return grouped.agg(**agg_kwargs)

    results = {}
    for m in metrics:
        col = m["column"]
        op = Aggregation(m["aggregation"])
        alias = m["alias"]
        series = df[col]
        if op == Aggregation.NUNIQUE:
            results[alias] = series.nunique()
        else:
            results[alias] = series.agg(op)
    return results


def _format_and_sort_results(
    raw_result: pd.DataFrame | dict[str, Any], sort: dict[str, str] | None, limit: int | None
) -> Any:
    """Applies sorting, limiting, and final formatting to the aggregated results."""
    if isinstance(raw_result, dict):
        return raw_result

    agg_df = raw_result

    if sort:
        sort_col = sort.get("column")
        direction = sort.get("direction", "asc")
        ascending = direction == "asc"

        valid_targets = set(agg_df.columns) | set(agg_df.index.names)
        if sort_col in valid_targets:
            agg_df = agg_df.sort_values(by=sort_col, ascending=ascending)  # type: ignore
        else:
            # Provide available targets for the correction chain
            clean_targets = sorted(list(valid_targets), key=lambda x: (x is not None, str(x)))

            raise ValueError(
                f"Sort column '{sort_col}' not found. Must be an aggregation alias or group dimension. "
                f"Available: {clean_targets}"
            )

    if limit is not None:
        agg_df = agg_df.head(limit)

    agg_df = agg_df.reset_index()
    agg_df.columns = [str(c) for c in agg_df.columns]

    return agg_df.to_dict(orient="records")


def analyze_data(
    df: pd.DataFrame,
    metrics: list[dict[str, Any]],
    filters: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    sort: dict[str, str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compiles and executes the Analytical Query IR against a DataFrame.

    Args:
        df: The source DataFrame.
        metrics: List of metric definitions (column, aggregation, alias).
        filters: Optional list of filter conditions.
        group_by: Optional list of columns to group by.
        sort: Optional sorting criteria (column, direction).
        limit: Optional row limit after sorting.

    Returns:
        A dictionary containing the 'result' key with the computed data.
    """
    logger.info(
        "Executing analyze_data | metrics: %d | grouped: %s | filtered: %s | sorted: %s | limit: %s",
        len(metrics),
        bool(group_by),
        bool(filters),
        bool(sort),
        limit,
    )

    # 1. Validation (Fast-failing with actionable context for the LLM correction loop)
    seen_aliases = set()
    available_cols = list(df.columns)

    for m in metrics:
        if m["aggregation"] not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation '{m['aggregation']}'. Choose from: {SUPPORTED_OPERATIONS}")
        if m["column"] not in df.columns:
            # Inject available columns so the LLM can self-heal
            raise ValueError(f"Metric target column '{m['column']}' not found in dataset. Available: {available_cols}")
        if m["alias"] in seen_aliases:
            raise ValueError(f"Duplicate metric alias '{m['alias']}'. Each metric must have a unique alias.")
        seen_aliases.add(m["alias"])

    if group_by:
        for col in group_by:
            if col not in df.columns:
                # Inject available columns so the LLM can self-heal
                raise ValueError(f"Group-by column '{col}' not found in dataset. Available: {available_cols}")

    if filters is None:
        filters = []

    # 2. Filtering phase
    working_df = apply_filters(df, filters) if filters else df.copy()

    if working_df.empty:
        return {"result": None, "warning": "Filtered dataset is empty."}

    # 3. Execution phase (Aggregation)
    raw_result = _perform_aggregation(working_df, metrics, group_by)

    # 4. Assembly phase (Sort, Limit, Formatting)
    final_result = _format_and_sort_results(raw_result, sort, limit)

    return {"result": final_result}

"""Shared row-filtering engine.

Used by the analyze_data tool and the filter_dataset wrangling tool.
Ensures a single, deterministic filter grammar and implementation across
all data subsetting operations.
"""

from typing import Any

import pandas as pd

from ...contracts import FilterOperator


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """Applies a sequence of strict filter conditions to a DataFrame.

    Args:
        df: The source DataFrame to filter.
        filters: A list of filter dictionaries containing 'column', 'operator', and 'value'.

    Returns:
        A new DataFrame containing only the rows that satisfy all conditions.

    Raises:
        ValueError: If a filter specification is malformed, references a missing column,
                    or uses an unsupported operator/value combination.
    """
    working_df = df.copy()

    for f in filters:
        # Validate filter structure
        if not all(k in f for k in ("column", "operator", "value")):
            raise ValueError(f"Invalid filter specification (missing keys): {f}")

        col = f["column"]
        raw_op = f["operator"]
        val = f["value"]

        # Validate column existence
        if col not in working_df.columns:
            raise ValueError(f"Filter column '{col}' not found. Available: {list(working_df.columns)}")

        # Validate and resolve operator
        try:
            op = FilterOperator(raw_op)
        except ValueError:
            valid_ops = [o.value for o in FilterOperator]
            # Suppress the original traceback to keep error logs clean for the LLM
            raise ValueError(f"Unsupported filter operator: '{raw_op}'. Choose from: {valid_ops}") from None

        # Handle None values explicitly for equality/inequality
        if val is None:
            if op == FilterOperator.EQ:
                working_df = working_df[working_df[col].isna()]
            elif op == FilterOperator.NEQ:
                working_df = working_df[working_df[col].notna()]
            else:
                raise ValueError(f"Operator '{op.value}' with None value is not supported. Use '==' or '!='.")
            continue

        # Dispatch filtering logic based on operator
        if op == FilterOperator.EQ:
            working_df = working_df[working_df[col] == val]
        elif op == FilterOperator.NEQ:
            working_df = working_df[working_df[col] != val]
        elif op == FilterOperator.GT:
            working_df = working_df[working_df[col] > val]
        elif op == FilterOperator.LT:
            working_df = working_df[working_df[col] < val]
        elif op == FilterOperator.GTE:
            working_df = working_df[working_df[col] >= val]
        elif op == FilterOperator.LTE:
            working_df = working_df[working_df[col] <= val]
        elif op == FilterOperator.IN:
            if not isinstance(val, list):
                raise ValueError(f"Operator 'in' requires a list of values, got {type(val).__name__}.")
            working_df = working_df[working_df[col].isin(val)]
        elif op == FilterOperator.NOT_IN:
            if not isinstance(val, list):
                raise ValueError(f"Operator 'not in' requires a list of values, got {type(val).__name__}.")
            working_df = working_df[~working_df[col].isin(val)]
        elif op == FilterOperator.CONTAINS:
            working_df = working_df[working_df[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op == FilterOperator.STARTSWITH:
            working_df = working_df[working_df[col].astype(str).str.startswith(str(val), na=False)]
        elif op == FilterOperator.ENDSWITH:
            working_df = working_df[working_df[col].astype(str).str.endswith(str(val), na=False)]

    return working_df

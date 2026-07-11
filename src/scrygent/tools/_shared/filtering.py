"""Shared row-filtering engine. Used by analyze_data.py (Tier 1 query
engine) and wrangling.py (filter_dataset). One filter grammar, one
implementation -- both callers stay behaviorally identical by construction."""

from typing import Any
import pandas as pd

from ...contracts import FilterOperator


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    working_df = df.copy()

    for f in filters:
        if not all(k in f for k in ("column", "operator", "value")):
            raise ValueError(f"Invalid filter specification (missing keys): {f}")

        col = f["column"]
        raw_op = f["operator"]
        try:
            op = FilterOperator(raw_op)
        except ValueError:
            raise ValueError(
                f"Unsupported filter operator: '{raw_op}'. "
                f"Choose from: {[o.value for o in FilterOperator]}"
            ) from None
        val = f["value"]

        if col not in working_df.columns:
            raise ValueError(f"Filter column '{col}' not found. Available: {list(working_df.columns)}")

        if val is None:
            if op == FilterOperator.EQ:
                working_df = working_df[working_df[col].isna()]
            elif op == FilterOperator.NEQ:
                working_df = working_df[working_df[col].notna()]
            else:
                raise ValueError(f"Operator '{op}' with None value is not supported. Use '==' or '!='.")
            continue

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
                raise ValueError(f"Operator 'in' requires a list of values, got {type(val)}.")
            working_df = working_df[working_df[col].isin(val)]
        elif op == FilterOperator.NOT_IN:
            if not isinstance(val, list):
                raise ValueError(f"Operator 'not in' requires a list of values, got {type(val)}.")
            working_df = working_df[~working_df[col].isin(val)]
        elif op == FilterOperator.CONTAINS:
            working_df = working_df[working_df[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op == FilterOperator.STARTSWITH:
            working_df = working_df[working_df[col].astype(str).str.startswith(str(val), na=False)]
        elif op == FilterOperator.ENDSWITH:
            working_df = working_df[working_df[col].astype(str).str.endswith(str(val), na=False)]
        # No trailing else needed: op is guaranteed to be a valid FilterOperator
        # member at this point (invalid raw values are caught above), so every
        # enum member is exhaustively handled by the branches above.

    return working_df

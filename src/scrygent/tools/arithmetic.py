"""Deterministic arithmetic engine using numexpr.

Provides safe, row-wise cross-column mathematics and scalar metric
evaluation without exposing Python's eval() or arbitrary code execution.
"""

import logging
import re
from pathlib import Path
from typing import Any

import numexpr as ne  # type: ignore
import pandas as pd

from .io import load_csv, write_temp_csv

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_NUMEXPR_FUNCTIONS = {
    "sqrt",
    "abs",
    "exp",
    "log",
    "log10",
    "log2",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "where",
}


def _extract_column_refs(expression: str, df_columns: set[str]) -> set[str]:
    """Extracts valid DataFrame column references from a numexpr expression.

    Raises a ValueError if the expression references unknown identifiers
    that are neither valid columns nor recognized numexpr functions.
    """
    identifiers = set(_IDENTIFIER_PATTERN.findall(expression))
    unknown = identifiers - df_columns - _NUMEXPR_FUNCTIONS
    if unknown:
        raise ValueError(
            f"Expression references unknown identifier(s): {sorted(unknown)}. Available columns: {sorted(df_columns)}"
        )
    return identifiers & df_columns


def _require_numeric(df: pd.DataFrame, columns: set[str]) -> None:
    """Validates that all referenced columns are numeric."""
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expression references non-numeric column(s): {non_numeric}")


def derive_column(
    current_csv_path: Path,
    new_column: str,
    expression: str,
) -> dict[str, Any]:
    """Evaluates a numexpr expression row-wise and writes the result as a new column.

    Args:
        current_csv_path: Path to the active CSV dataset.
        new_column: Name of the new column to create.
        expression: The numexpr-compatible mathematical expression.

    Returns:
        A dictionary containing the new CSV path, column name, expression,
        and a sample of the new column's statistics.
    """
    if not expression.strip():
        raise ValueError("derive_column requires a non-empty expression.")

    logger.info("Executing derive_column | new_column: %s | expression: %s", new_column, expression)

    df = load_csv(current_csv_path)

    if new_column in df.columns:
        raise ValueError(f"Column '{new_column}' already exists. Choose a distinct name.")

    referenced_columns = _extract_column_refs(expression, set(df.columns))
    if not referenced_columns:
        raise ValueError("Expression must reference at least 1 existing column.")
    _require_numeric(df, referenced_columns)

    local_dict = {col: df[col].to_numpy() for col in referenced_columns}

    try:
        # global_dict={} explicitly denies numexpr any fallback to the calling frame's globals.
        result = ne.evaluate(expression, local_dict=local_dict, global_dict={})
    except Exception as e:
        # Suppress the numexpr traceback to keep the LLM correction chain clean.
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from None

    df[new_column] = result
    new_path = write_temp_csv(df, prefix="scrygent_derive_")

    return {
        "current_csv_path": str(new_path),
        "new_column": new_column,
        "expression": expression,
        "sample": {
            "min": df[new_column].min(),
            "max": df[new_column].max(),
            "mean": df[new_column].mean(),
        },
    }


def evaluate_metrics(
    expression: str,
    values: dict[str, float],
) -> dict[str, Any]:
    """Evaluates a numexpr expression over a dictionary of pre-computed scalar values.

    Args:
        expression: The numexpr-compatible mathematical expression.
        values: Dictionary of named scalar values to inject into the expression.

    Returns:
        A dictionary containing the expression and the computed float result.
    """
    if not expression.strip():
        raise ValueError("evaluate_metrics requires a non-empty expression.")
    if not values:
        raise ValueError("evaluate_metrics requires at least 1 named value.")

    logger.info("Executing evaluate_metrics | expression: %s | inputs: %s", expression, list(values.keys()))

    referenced = _extract_column_refs(expression, set(values.keys()))
    if not referenced:
        raise ValueError("Expression must reference at least 1 provided value.")

    local_dict = {k: float(v) for k, v in values.items() if k in referenced}

    try:
        result = ne.evaluate(expression, local_dict=local_dict, global_dict={})
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from None

    return {"expression": expression, "result": float(result)}

import logging
import re
from pathlib import Path
from typing import Any

import numexpr as ne
import pandas as pd

from .io import load_csv, write_temp_csv

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# numexpr's built-in function names -- these appear as identifiers in an
# expression (e.g. "sqrt(Revenue)") but are not column references. Kept
# as an explicit whitelist so we can tell "identifier that must be a
# column" apart from "identifier that's a numexpr function" without
# guessing.
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
    """Extracts identifiers from the expression that correspond to real
    DataFrame columns, excluding numexpr's own function names. Any
    identifier that is neither a known column nor a numexpr function is
    an error -- this is what stops the expression from silently reading
    stray names (or in principle, referencing something it shouldn't).
    """
    identifiers = set(_IDENTIFIER_PATTERN.findall(expression))
    unknown = identifiers - df_columns - _NUMEXPR_FUNCTIONS
    if unknown:
        raise ValueError(
            f"Expression references unknown identifier(s): {sorted(unknown)}. Available columns: {sorted(df_columns)}"
        )
    return identifiers & df_columns


def _require_numeric(df: pd.DataFrame, columns: set[str]) -> None:
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expression references non-numeric column(s): {non_numeric}")


# --- derive_column: row-wise cross-column math, materialized as a new column ---
def derive_column(
    current_csv_path: Path,
    new_column: str,
    expression: str,
) -> dict[str, Any]:
    """Evaluates a numexpr expression row-wise across existing numeric
    columns and writes the result as a new column, e.g.
    expression="Revenue - Cost" with new_column="Profit".

    Writes the transformed dataset to a new temp CSV and updates
    current_csv_path, following the same wrangling-tool convention as
    filter_dataset/normalize_column (never overwrites original_csv_path).

    Returns:
        {"current_csv_path": str, "new_column": str, "expression": str,
         "sample": {"min": ..., "max": ..., "mean": ...}}
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
        # global_dict={} -- explicitly denies numexpr any fallback to the
        # calling frame's globals, so the only names resolvable are the
        # ones we put in local_dict.
        result = ne.evaluate(expression, local_dict=local_dict, global_dict={})
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e

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


# --- evaluate_metrics: scalar math over already-computed step_outputs values ---


def evaluate_metrics(
    expression: str,
    values: dict[str, float],
) -> dict[str, Any]:
    """Evaluates a numexpr expression over a small dict of already-computed
    scalar values (e.g. two prior analyze_data results), rather than
    over a DataFrame. This covers the common evals pattern of deriving a
    ratio/delta from two previous steps' aggregates without re-reading
    the CSV or risking a fresh hallucinated computation --
    e.g. values={"total_profit": 42000, "total_revenue": 150000},
    expression="total_profit / total_revenue" -> profit margin.

    Same safety guarantee as derive_column: numexpr, explicit
    local_dict, no global_dict fallback.

    Returns:
        {"expression": str, "result": float}
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
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e

    return {"expression": expression, "result": float(result)}

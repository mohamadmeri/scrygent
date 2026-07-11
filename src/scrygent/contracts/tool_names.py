"""Closed vocabulary for all deterministic tool identifiers.

This module defines the exact string literals used by the Planner to
route execution steps to the deterministic compiler. The LLM must emit
these exact values to trigger tool dispatch.
"""

from enum import StrEnum


class ToolName(StrEnum):
    """Registered identifiers for the deterministic execution engine."""

    ANALYZE_DATA = "analyze_data"
    FILTER_DATASET = "filter_dataset"
    NORMALIZE_COLUMN = "normalize_column"
    RESET_DATASET = "reset_dataset"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    DETECT_OUTLIERS = "detect_outliers"
    REQUEST_COLUMN_STATS = "request_column_stats"
    GENERATE_PLOT = "generate_plot"
    DERIVE_COLUMN = "derive_column"
    EVALUATE_METRICS = "evaluate_metrics"

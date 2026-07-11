"""Enumerations for the analyze_data tool family."""

from enum import StrEnum


class Aggregation(StrEnum):
    """Supported aggregation operations for analytical queries."""

    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    NUNIQUE = "nunique"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"

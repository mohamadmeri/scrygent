"""Enums owned by the analyze_data tool family."""

from enum import StrEnum


class Aggregation(StrEnum):
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    NUNIQUE = "nunique"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"

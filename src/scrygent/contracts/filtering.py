"""Shared filter-operator vocabulary for the compiler pipeline.

This module defines the closed set of comparison operators used by both
the analytical query engine and the data wrangling tools. Centralizing
these operators ensures strict schema alignment across filtering contexts.
"""

from enum import StrEnum


class FilterOperator(StrEnum):
    """Supported comparison operators for row-level filtering."""

    EQ = "=="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    IN = "in"
    NOT_IN = "not in"
    CONTAINS = "contains"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"

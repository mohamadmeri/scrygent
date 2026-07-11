"""Shared filter-operator vocabulary. Consumed by analyze_data's
AnalyzeDataParams.filters and wrangling's FilterDatasetParams.filters --
both tools filter on the same operator set, so it's defined once here
rather than owned by either tool family."""

from enum import StrEnum


class FilterOperator(StrEnum):
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

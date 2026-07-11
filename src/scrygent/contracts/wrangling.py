"""Enumerations for the data wrangling tool family."""

from enum import StrEnum


class NormalizeMethod(StrEnum):
    """Supported transformation methods for column normalization."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    LOG = "log"
    STRIP = "strip"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    TITLE_CASE = "title_case"

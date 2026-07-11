"""Enums owned by the wrangling tool family."""

from enum import StrEnum


class NormalizeMethod(StrEnum):
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    LOG = "log"
    STRIP = "strip"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    TITLE_CASE = "title_case"

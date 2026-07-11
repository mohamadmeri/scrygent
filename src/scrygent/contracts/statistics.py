"""Enums owned by the statistics tool family."""

from enum import StrEnum


class CorrelationMethod(StrEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class RegressionMethod(StrEnum):
    LINEAR = "linear"


class OutlierMethod(StrEnum):
    IQR = "iqr"
    Z_SCORE = "z_score"

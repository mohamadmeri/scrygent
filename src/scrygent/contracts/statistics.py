"""Enumerations for the statistics tool family."""

from enum import StrEnum


class CorrelationMethod(StrEnum):
    """Supported algorithms for computing column correlations."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class RegressionMethod(StrEnum):
    """Supported algorithms for computing linear regressions."""

    LINEAR = "linear"


class OutlierMethod(StrEnum):
    """Supported algorithms for detecting statistical outliers."""

    IQR = "iqr"
    Z_SCORE = "z_score"

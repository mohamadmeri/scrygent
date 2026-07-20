"""Intermediate Representation for statistical analysis operations.

Defines the strict payloads for correlation, regression, outlier detection,
and on-demand column statistics fetching.
"""

from pydantic import Field, model_validator

from ..base_model import ScrygentBaseModel
from ..contracts import CorrelationMethod, OutlierMethod, RegressionMethod


class CorrelationParams(ScrygentBaseModel):
    """IR for computing correlation between two or more columns."""

    columns: list[str] = Field(min_length=2, description="The columns to compute correlation for.")
    method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON, description="The correlation algorithm to use.")


class RegressionParams(ScrygentBaseModel):
    """IR for computing linear regression."""

    target: str = Field(description="The target column for the regression.")
    features: list[str] = Field(min_length=1, description="The feature columns for the regression.")
    method: RegressionMethod = Field(default=RegressionMethod.LINEAR, description="The regression algorithm to use.")

    @model_validator(mode="after")
    def _target_not_in_features(self) -> RegressionParams:
        """Ensures the target column is not also listed as a feature."""
        if self.target in self.features:
            raise ValueError(f"target '{self.target}' cannot also appear in features.")
        return self


class OutlierParams(ScrygentBaseModel):
    """IR for detecting statistical outliers in a single column."""

    column: str = Field(description="The column to analyze for outliers.")
    method: OutlierMethod = Field(default=OutlierMethod.IQR, description="The outlier detection algorithm to use.")


class ColumnStatsParams(ScrygentBaseModel):
    """IR for fetching detailed statistical metrics for specific columns."""

    columns: list[str] = Field(min_length=1, description="The columns to compute detailed statistics for.")

from pydantic import Field, model_validator

from ..base_model import ScrygentBaseModel
from ..contracts import (
    CorrelationMethod,
    RegressionMethod,
    OutlierMethod,
)


class CorrelationParams(ScrygentBaseModel):
    columns: list[str] = Field(min_length=2)
    method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON)


class RegressionParams(ScrygentBaseModel):
    target: str
    features: list[str] = Field(min_length=1)
    method: RegressionMethod = Field(default=RegressionMethod.LINEAR)

    @model_validator(mode="after")
    def _target_not_in_features(self) -> "RegressionParams":
        if self.target in self.features:
            raise ValueError(f"target '{self.target}' cannot also appear in features.")
        return self


class OutlierParams(ScrygentBaseModel):
    column: str
    method: OutlierMethod = Field(default=OutlierMethod.IQR)


class ColumnStatsParams(ScrygentBaseModel):
    columns: list[str] = Field(min_length=1)

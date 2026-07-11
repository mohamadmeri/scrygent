from pydantic import Field

from ..base_model import ScrygentBaseModel


class DeriveColumnParams(ScrygentBaseModel):
    new_column: str = Field(min_length=1)
    expression: str = Field(min_length=1)


class EvaluateMetricsParams(ScrygentBaseModel):
    expression: str = Field(min_length=1)
    values: dict[str, float] = Field(min_length=1)

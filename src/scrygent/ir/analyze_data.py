from typing import Literal

from pydantic import Field, model_validator

from .filtering import FilterCondition
from ..base_model import ScrygentBaseModel
from ..contracts import Aggregation


class Metric(ScrygentBaseModel):
    column: str = Field(min_length=1, description="The exact name of the column to aggregate.")
    aggregation: Aggregation
    alias: str = Field(min_length=1, description="Name for the output key (e.g., 'Total Sales').")


class SortCondition(ScrygentBaseModel):
    column: str = Field(min_length=1, description="Column or metric alias to sort by.")
    direction: Literal["asc", "desc"]


class AnalyzeDataParams(ScrygentBaseModel):
    """IR for analyze_data. Filter -> Group -> Aggregate -> Sort -> Limit."""
    filters: list[FilterCondition] | None = Field(default=None)
    group_by: list[str] | None = Field(default=None, description="Columns to GROUP BY.")
    metrics: list[Metric] = Field(min_length=1, description="The mathematical aggregations to compute.")
    sort: SortCondition | None = Field(default=None)
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _aliases_unique_and_sort_resolvable(self) -> "AnalyzeDataParams":
        aliases = [m.alias for m in self.metrics]
        if len(aliases) != len(set(aliases)):
            dupes = sorted({a for a in aliases if aliases.count(a) > 1})
            raise ValueError(
                f"Duplicate metric alias(es): {dupes}. Each metric's alias must be "
                "unique -- duplicates silently collide in the output record."
            )

        if self.sort is not None:
            valid_targets = set(aliases) | set(self.group_by or [])
            if self.sort.column not in valid_targets:
                raise ValueError(
                    f"sort.column '{self.sort.column}' is not a metric alias or "
                    f"group_by column produced by this plan. Valid options: "
                    f"{sorted(valid_targets)}."
                )
        return self

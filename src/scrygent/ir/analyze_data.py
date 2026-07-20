"""Intermediate Representation for the unified analytical query engine.

This module defines the strict Pydantic schema for the `analyze_data` tool,
which consolidates filtering, grouping, aggregation, sorting, and limiting
into a single declarative payload.
"""

from typing import Literal

from pydantic import Field, model_validator

from ..base_model import ScrygentBaseModel
from ..contracts import Aggregation
from .filtering import FilterCondition


class Metric(ScrygentBaseModel):
    """Defines a single mathematical aggregation to compute."""

    column: str = Field(min_length=1, description="The exact name of the column to aggregate.")
    aggregation: Aggregation = Field(description="The aggregation operation to apply.")
    alias: str = Field(min_length=1, description="The output key name for the computed metric.")


class SortCondition(ScrygentBaseModel):
    """Defines the sorting criteria for the final aggregated output."""

    column: str = Field(min_length=1, description="The column or metric alias to sort by.")
    direction: Literal["asc", "desc"] = Field(description="The sort direction.")


class AnalyzeDataParams(ScrygentBaseModel):
    """IR for the unified analytical query tool.

    Enforces the execution pipeline: Filter -> Group -> Aggregate -> Sort -> Limit.
    """

    filters: list[FilterCondition] | None = Field(default=None, description="Row-level filtering conditions.")
    group_by: list[str] | None = Field(default=None, description="Columns to group the data by.")
    metrics: list[Metric] | None = Field(default=None, description="The mathematical aggregations to compute.")
    sort: SortCondition | None = Field(default=None, description="Optional sorting applied to the aggregated result.")
    limit: int | None = Field(default=None, ge=1, description="Optional row limit applied after sorting.")

    @model_validator(mode="after")
    def _aliases_unique_and_sort_resolvable(self) -> AnalyzeDataParams:
        """Validates that metric aliases are unique and the sort target is resolvable."""
        aliases = []
        if self.metrics is not None:
            aliases = [m.alias for m in self.metrics]
            if len(aliases) != len(set(aliases)):
                dupes = sorted({a for a in aliases if aliases.count(a) > 1})
                raise ValueError(
                    f"Duplicate metric alias(es): {dupes}. Each metric's alias must be unique to prevent silent collisions in the output record."
                )

        if self.sort is not None:
            # If we are aggregating or grouping, we MUST sort by the output columns.
            # If we are just sorting raw data (no metrics/groups), we skip this IR check
            # and let the Python tool validate the column against df.columns at runtime.
            if self.metrics is not None or self.group_by is not None:
                valid_targets = set(aliases) | set(self.group_by or [])
                if self.sort.column not in valid_targets:
                    raise ValueError(
                        f"sort.column '{self.sort.column}' is not a metric alias or "
                        f"group_by column produced by this plan. Valid options: "
                        f"{sorted(valid_targets)}."
                    )
        return self

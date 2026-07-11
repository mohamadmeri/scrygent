"""Intermediate Representation for data wrangling operations.

Defines the strict payloads for dataset filtering, column normalization,
and dataset resetting.
"""

from pydantic import Field

from ..base_model import ScrygentBaseModel
from ..contracts import NormalizeMethod
from .filtering import FilterCondition


class FilterDatasetParams(ScrygentBaseModel):
    """IR for filtering the dataset and writing the result to a new temporary CSV."""

    filters: list[FilterCondition] = Field(min_length=1, description="The conditions to filter the dataset by.")


class NormalizeColumnParams(ScrygentBaseModel):
    """IR for applying a transformation method to a specific column."""

    column: str = Field(description="The column to normalize.")
    method: NormalizeMethod = Field(description="The normalization method to apply.")


class NoParams(ScrygentBaseModel):
    """IR for tools that require no LLM-supplied parameters.

    Currently used exclusively by `reset_dataset`, which consumes
    `AgentState.original_csv_path` directly. The Planner must emit an
    empty dictionary for these steps to satisfy the strict schema boundary.
    """

    pass

"""Intermediate Representation for arithmetic and derived column operations.

These schemas define the payloads for safe, deterministic mathematical
evaluations using `numexpr`, ensuring no arbitrary code execution occurs.
"""

from pydantic import Field

from ..base_model import ScrygentBaseModel


class DeriveColumnParams(ScrygentBaseModel):
    """IR for creating a new column via a safe mathematical expression."""

    new_column: str = Field(min_length=1, description="The name of the new column to create.")
    expression: str = Field(min_length=1, description="The numexpr-compatible mathematical expression.")


class EvaluateMetricsParams(ScrygentBaseModel):
    """IR for evaluating a standalone mathematical expression against scalar values."""

    expression: str = Field(min_length=1, description="The numexpr-compatible mathematical expression.")
    values: dict[str, float] = Field(min_length=1, description="Scalar variables to inject into the expression.")

"""Shared filter-operator vocabulary and condition shapes for the IR layer.

This module defines the discriminated union for row-level filtering.
Consumed by analyze_data and filter_dataset, it ensures that the
operator and value shape are strictly aligned at the schema boundary.
"""

from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, Tag

from ..base_model import ScrygentBaseModel
from ..contracts import FilterOperator

_SCALAR_OPS = (
    FilterOperator.EQ,
    FilterOperator.NEQ,
    FilterOperator.GT,
    FilterOperator.LT,
    FilterOperator.GTE,
    FilterOperator.LTE,
)
_LIST_OPS = (FilterOperator.IN, FilterOperator.NOT_IN)
_STRING_OPS = (FilterOperator.CONTAINS, FilterOperator.STARTSWITH, FilterOperator.ENDSWITH)


class ScalarFilterCondition(ScrygentBaseModel):
    """Filter condition for scalar comparisons (==, !=, >, <, >=, <=)."""

    column: str = Field(min_length=1, description="The column to filter.")
    operator: Literal[_SCALAR_OPS] = Field(description="The scalar comparison operator.")  # type: ignore
    value: str | int | float | bool = Field(description="The scalar value to compare against.")


class ListFilterCondition(ScrygentBaseModel):
    """Filter condition for list membership (in, not in)."""

    column: str = Field(min_length=1, description="The column to filter.")
    operator: Literal[_LIST_OPS] = Field(description="The list membership operator.")  # type: ignore
    value: list[str | int | float | bool] = Field(
        min_length=1, description="The list of values to test membership against."
    )


class StringFilterCondition(ScrygentBaseModel):
    """Filter condition for string matching (contains, startswith, endswith)."""

    column: str = Field(min_length=1, description="The column to filter.")
    operator: Literal[_STRING_OPS] = Field(description="The string matching operator.")  # type: ignore
    value: str = Field(min_length=1, description="The string pattern to match.")


def _filter_tag(v: Any) -> str:
    """Routes each filter payload to its specific shape based on the operator.

    This discriminator function ensures that the Pydantic validator selects
    the correct branch model before validating the value type and shape.
    """
    op = v.get("operator") if isinstance(v, dict) else getattr(v, "operator", None)
    if op in _SCALAR_OPS:
        return "scalar"
    if op in _LIST_OPS:
        return "list"
    if op in _STRING_OPS:
        return "string"
    raise ValueError(
        f"Unrecognized filter operator: {op!r}. Expected one of: {sorted(m.value for m in FilterOperator)}."
    )


FilterCondition = Annotated[
    Annotated[ScalarFilterCondition, Tag("scalar")]
    | Annotated[ListFilterCondition, Tag("list")]
    | Annotated[StringFilterCondition, Tag("string")],
    Discriminator(_filter_tag),
]

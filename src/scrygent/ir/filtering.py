"""Shared filter-operator vocabulary and condition shapes. Consumed by
analyze_data's AnalyzeDataParams.filters and wrangling's
FilterDatasetParams.filters -- both tools filter on the same operator
set, so it's defined once here rather than owned by either tool family.

FilterCondition is a discriminated union, not a single model. Each
FilterOperator implies a specific value shape (scalar comparison, list
membership, or string matching), and a flat `value: Any` field let any
operator pair with any value type -- e.g. operator=">" with a list
value -- passing Pydantic cleanly and failing later inside the tool as
an unlabeled TypeError. Tagging on operator group makes that
combination unrepresentable instead of merely unlikely.
"""

from typing import Annotated, Any, Literal, Union

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
    """==, !=, >, <, >=, <= -- compares a column against one scalar."""
    column: str = Field(min_length=1)
    operator: Literal[_SCALAR_OPS] # type: ignore
    value: str | int | float | bool


class ListFilterCondition(ScrygentBaseModel):
    """in, not in -- tests column membership against a non-empty list."""
    column: str = Field(min_length=1)
    operator: Literal[_LIST_OPS] # type: ignore
    value: list[str | int | float | bool] = Field(min_length=1)


class StringFilterCondition(ScrygentBaseModel):
    """contains, startswith, endswith -- string ops need a non-empty string."""
    column: str = Field(min_length=1)
    operator: Literal[_STRING_OPS] # type: ignore
    value: str = Field(min_length=1)


def _filter_tag(v: Any) -> str:
    """Routes each payload to its shape based on operator, before the
    per-branch models validate value type/shape."""
    op = v.get("operator") if isinstance(v, dict) else getattr(v, "operator", None)
    if op in _SCALAR_OPS:
        return "scalar"
    if op in _LIST_OPS:
        return "list"
    if op in _STRING_OPS:
        return "string"
    raise ValueError(
        f"Unrecognized filter operator: {op!r}. "
        f"Expected one of: {sorted(m.value for m in FilterOperator)}."
    )


FilterCondition = Annotated[
    Union[
        Annotated[ScalarFilterCondition, Tag("scalar")],
        Annotated[ListFilterCondition, Tag("list")],
        Annotated[StringFilterCondition, Tag("string")],
    ],
    Discriminator(_filter_tag),
]

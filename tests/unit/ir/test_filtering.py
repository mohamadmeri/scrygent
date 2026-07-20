"""Destructive test suite for the filtering Intermediate Representation models.

This module aggressively tests the discriminated union of filter conditions.
It ensures that operator/value pairs are strictly aligned, hallucinated
operators are rejected, and boundary-polluting NumPy types are scrubbed
before reaching the deterministic execution engine.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from scrygent.ir.filtering import FilterCondition, ListFilterCondition, ScalarFilterCondition, StringFilterCondition


class TestScalarFilterCondition:
    """Tests validating scalar comparison filters (==, !=, >, <, >=, <=)."""

    def test_accepts_valid_scalar_payload_and_scrubs_numpy(self) -> None:
        """Inject a valid scalar payload containing a NumPy integer.

        Asserts the model accepts the payload and the boundary scrubs the
        `np.int64` into a native Python `int`.
        """
        payload: dict[str, Any] = {
            "column": "age",
            "operator": ">",
            "value": np.int64(30),
        }
        model = ScalarFilterCondition(**payload)

        assert model.column == "age"
        assert model.value == 30
        assert isinstance(model.value, int)
        assert not isinstance(model.value, np.integer)

    def test_rejects_list_value_for_scalar_operator(self) -> None:
        """Inject a list value alongside a scalar operator like '>'.

        The schema must enforce strict type alignment, rejecting lists for
        scalar comparisons to prevent Pandas evaluation errors.
        """
        payload: dict[str, Any] = {
            "column": "age",
            "operator": ">",
            "value": [30, 40],
        }
        with pytest.raises(ValidationError) as exc_info:
            ScalarFilterCondition(**payload)

        # Pydantic v2 emits multiple errors for union mismatches.
        assert "value" in str(exc_info.value)
        assert "Input should be a valid string" in str(exc_info.value)
        assert "Input should be a valid integer" in str(exc_info.value)

    def test_rejects_empty_column_name(self) -> None:
        """Inject an empty string for the `column` field.

        The schema enforces `min_length=1` to prevent blind filtering on
        non-existent columns.
        """
        payload: dict[str, Any] = {"column": "", "operator": ">", "value": 30}
        with pytest.raises(ValidationError) as exc_info:
            ScalarFilterCondition(**payload)

        assert "String should have at least 1 character" in str(exc_info.value)


class TestListFilterCondition:
    """Tests validating list membership filters (in, not in)."""

    def test_accepts_valid_list_payload(self) -> None:
        """Verify a baseline valid list payload passes schema validation."""
        payload: dict[str, Any] = {
            "column": "city",
            "operator": "in",
            "value": ["NYC", "LA"],
        }
        model = ListFilterCondition(**payload)

        assert model.operator == "in"
        assert model.value == ["NYC", "LA"]

    def test_rejects_empty_list_value(self) -> None:
        """Inject an empty list for the `value` field.

        The schema enforces `min_length=1` because an empty `in`/`not in`
        list is a no-op that wastes execution cycles.
        """
        payload: dict[str, Any] = {
            "column": "city",
            "operator": "in",
            "value": [],
        }
        with pytest.raises(ValidationError) as exc_info:
            ListFilterCondition(**payload)

        assert "List should have at least 1 item" in str(exc_info.value)

    def test_rejects_scalar_value_for_list_operator(self) -> None:
        """Inject a scalar value alongside a list operator like 'in'.

        The schema must reject scalars for list operators to prevent iteration
        errors in the executor.
        """
        payload: dict[str, Any] = {
            "column": "city",
            "operator": "in",
            "value": "NYC",
        }
        with pytest.raises(ValidationError) as exc_info:
            ListFilterCondition(**payload)

        assert "Input should be a valid list" in str(exc_info.value)


class TestStringFilterCondition:
    """Tests validating string matching filters (contains, startswith, endswith)."""

    def test_accepts_valid_string_payload(self) -> None:
        """Verify a baseline valid string payload passes schema validation."""
        payload: dict[str, Any] = {
            "column": "name",
            "operator": "contains",
            "value": "John",
        }
        model = StringFilterCondition(**payload)

        assert model.operator == "contains"
        assert model.value == "John"

    def test_rejects_non_string_value_for_string_operator(self) -> None:
        """Inject an integer value alongside a string operator like 'contains'.

        The schema must enforce strict string matching to prevent Pandas
        AttributeError when calling `.str.contains()` on numeric data.
        """
        payload: dict[str, Any] = {
            "column": "name",
            "operator": "contains",
            "value": 123,
        }
        with pytest.raises(ValidationError) as exc_info:
            StringFilterCondition(**payload)

        assert "Input should be a valid string" in str(exc_info.value)


class TestFilterConditionUnion:
    """Tests validating the discriminated union routing logic of FilterCondition."""

    def test_union_routes_scalar_operator_correctly(self) -> None:
        """Verify the TypeAdapter correctly routes a scalar payload to ScalarFilterCondition."""
        adapter = TypeAdapter(FilterCondition)
        payload: dict[str, Any] = {"column": "x", "operator": "==", "value": 10}

        model = adapter.validate_python(payload)
        assert isinstance(model, ScalarFilterCondition)

    def test_union_rejects_hallucinated_operator_with_exact_error(self) -> None:
        """Inject a hallucinated operator like 'equals'.

        The discriminator function must raise an error halting execution,
        preventing the LLM from inventing new comparison logic.
        """
        adapter = TypeAdapter(FilterCondition)
        payload: dict[str, Any] = {"column": "x", "operator": "equals", "value": 10}

        # Pydantic v2 may propagate the raw ValueError from the discriminator
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            adapter.validate_python(payload)

        assert "Unrecognized filter operator: 'equals'" in str(exc_info.value)
        assert "Expected one of:" in str(exc_info.value)

    def test_union_rejects_missing_operator_field(self) -> None:
        """Inject a payload missing the `operator` field entirely.

        The discriminator must fail fast when the routing key is absent.
        """
        adapter = TypeAdapter(FilterCondition)
        payload: dict[str, Any] = {"column": "x", "value": 10}

        with pytest.raises((ValidationError, ValueError)) as exc_info:
            adapter.validate_python(payload)

        assert "Unrecognized filter operator: None" in str(exc_info.value)
        assert "Expected one of:" in str(exc_info.value)

    def test_union_rejects_extra_fields_in_union_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `malicious_field`.

        The `extra="forbid` rule must propagate through the discriminated union.
        """
        adapter = TypeAdapter(FilterCondition)
        payload: dict[str, Any] = {
            "column": "x",
            "operator": "==",
            "value": 10,
            "malicious_field": "fail",
        }

        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python(payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

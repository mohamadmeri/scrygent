"""Destructive test suite for the wrangling Intermediate Representation models.

This module aggressively tests the Pydantic IR schemas for filtering,
normalization, and parameterless tools. It ensures that hallucinated
methods, malformed payloads, and boundary-polluting types are strictly
rejected before reaching the deterministic execution engine.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scrygent.contracts.filtering import FilterOperator
from scrygent.contracts.wrangling import NormalizeMethod
from scrygent.ir.wrangling import FilterDatasetParams, NoParams, NormalizeColumnParams


class TestFilterDatasetParams:
    """Tests validating the strict schema and boundary enforcement of filter IR payloads."""

    def test_accepts_valid_filter_conditions_list(self) -> None:
        """Verify a baseline valid list of filter conditions passes schema validation.

        Ensures the primary execution path for data filtering is accepted.
        """
        payload: dict[str, Any] = {"filters": [{"column": "age", "operator": FilterOperator.GT, "value": 30}]}
        model = FilterDatasetParams(**payload)

        assert len(model.filters) == 1
        assert model.filters[0].column == "age"

    def test_rejects_empty_filters_list_with_min_length_error(self) -> None:
        """Inject an empty list for the filters field.

        The schema enforces `min_length=1` to prevent the LLM from emitting
        no-op filter steps that waste execution cycles.
        """
        payload: dict[str, Any] = {"filters": []}
        with pytest.raises(ValidationError) as exc_info:
            FilterDatasetParams(**payload)

        assert "List should have at least 1 item" in str(exc_info.value)

    def test_rejects_non_list_filters_payload(self) -> None:
        """Inject a dictionary instead of a list for the filters field.

        The contract must strictly enforce the list type to prevent iteration
        errors in the deterministic executor.
        """
        payload: dict[str, Any] = {"filters": {"column": "age", "operator": ">", "value": 30}}
        with pytest.raises(ValidationError) as exc_info:
            FilterDatasetParams(**payload)

        assert "Input should be a valid list" in str(exc_info.value)

    def test_scrubs_numpy_types_injected_into_filter_payload(self) -> None:
        """Inject a filter value containing a NumPy integer.

        The Hermetic JSON Boundary must intercept and scrub the `np.int64`
        to a native Python `int` before Pydantic freezes the model.
        """
        payload: dict[str, Any] = {"filters": [{"column": "age", "operator": FilterOperator.GT, "value": np.int64(30)}]}
        model = FilterDatasetParams(**payload)

        assert model.filters[0].value == 30
        assert isinstance(model.filters[0].value, int)
        assert not isinstance(model.filters[0].value, np.integer)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `malicious_field`.

        The `ScrygentBaseModel` configuration must strictly forbid extra keys
        to prevent schema drift.
        """
        payload: dict[str, Any] = {
            "filters": [{"column": "age", "operator": FilterOperator.GT, "value": 30}],
            "malicious_field": "should fail",
        }
        with pytest.raises(ValidationError) as exc_info:
            FilterDatasetParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestNormalizeColumnParams:
    """Tests validating the strict schema and closed vocabulary of normalization IR."""

    def test_accepts_valid_column_and_method(self) -> None:
        """Verify a baseline valid normalization payload passes schema validation."""
        payload: dict[str, Any] = {"column": "income", "method": NormalizeMethod.LOG}
        model = NormalizeColumnParams(**payload)

        assert model.column == "income"
        assert model.method == NormalizeMethod.LOG

    def test_rejects_hallucinated_normalization_method(self) -> None:
        """Inject an unsupported method string like 'standardize'.

        The schema must reject hallucinated methods to prevent attribute errors
        in the wrangling tool.
        """
        payload: dict[str, Any] = {"column": "income", "method": "standardize"}
        with pytest.raises(ValidationError) as exc_info:
            NormalizeColumnParams(**payload)

        assert "Input should be" in str(exc_info.value)
        assert "standardize" in str(exc_info.value)
        assert "min_max" in str(exc_info.value)

    def test_rejects_missing_required_fields(self) -> None:
        """Attempt to instantiate the model with missing `column` or `method`.

        Ensures strict failure when the LLM drops required parameters.
        """
        payload: dict[str, Any] = {"method": NormalizeMethod.LOG}
        with pytest.raises(ValidationError) as exc_info:
            NormalizeColumnParams(**payload)

        assert "Field required" in str(exc_info.value)
        assert "column" in str(exc_info.value)

    def test_rejects_non_string_column_identifier(self) -> None:
        """Inject an integer for the `column` field.

        The schema must enforce string-only column names to prevent type
        coercion bugs in Pandas indexing.
        """
        payload: dict[str, Any] = {"column": 12345, "method": NormalizeMethod.LOG}
        with pytest.raises(ValidationError) as exc_info:
            NormalizeColumnParams(**payload)

        assert "Input should be a valid string" in str(exc_info.value)


class TestNoParams:
    """Tests validating the strict empty schema for parameterless tools."""

    def test_accepts_empty_dict_payload(self) -> None:
        """Verify a baseline empty dictionary passes schema validation.

        Ensures tools like `reset_dataset` can be triggered without parameters.
        """
        model = NoParams()
        assert model.model_dump() == {}

    def test_rejects_any_injected_fields(self) -> None:
        """Inject a payload with an unexpected field.

        Even for parameterless tools, the `extra="forbid"` rule must apply
        to prevent the LLM from attaching hallucinated configuration.
        """
        payload: dict[str, Any] = {"malicious_field": "should fail"}
        with pytest.raises(ValidationError) as exc_info:
            NoParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

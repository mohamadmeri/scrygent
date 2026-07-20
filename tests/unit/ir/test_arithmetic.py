"""Destructive test suite for the arithmetic Intermediate Representation models.

This module aggressively tests the Pydantic IR schemas for derived columns
and scalar metric evaluation. It ensures that empty expressions, missing
variables, and boundary-polluting NumPy types are strictly rejected before
reaching the safe `numexpr` execution engine.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scrygent.ir.arithmetic import DeriveColumnParams, EvaluateMetricsParams


class TestDeriveColumnParams:
    """Tests validating the strict schema for derived column expressions."""

    def test_accepts_valid_payload_and_scrubs_numpy_str(self) -> None:
        """Inject a valid payload containing a NumPy string for `new_column`.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.str_` into a native Python `str`.
        """
        payload: dict[str, Any] = {
            "new_column": np.str_("total_fare"),
            "expression": "fare + taxes",
        }
        model = DeriveColumnParams(**payload)

        assert model.new_column == "total_fare"
        assert isinstance(model.new_column, str)
        assert not isinstance(model.new_column, np.str_)

    def test_rejects_empty_new_column_string(self) -> None:
        """Inject an empty string for the `new_column` field.

        The schema enforces `min_length=1` to prevent the LLM from omitting
        the target column name while satisfying the type requirement.
        """
        payload: dict[str, Any] = {"new_column": "", "expression": "fare + taxes"}
        with pytest.raises(ValidationError) as exc_info:
            DeriveColumnParams(**payload)

        assert "String should have at least 1 character" in str(exc_info.value)

    def test_rejects_empty_expression_string(self) -> None:
        """Inject an empty string for the `expression` field.

        The schema enforces `min_length=1` to prevent empty `numexpr` evaluations.
        """
        payload: dict[str, Any] = {"new_column": "total_fare", "expression": ""}
        with pytest.raises(ValidationError) as exc_info:
            DeriveColumnParams(**payload)

        assert "String should have at least 1 character" in str(exc_info.value)

    def test_rejects_missing_required_fields(self) -> None:
        """Attempt to instantiate the model with missing `expression`.

        Ensures strict failure when the LLM drops required parameters.
        """
        payload: dict[str, Any] = {"new_column": "total_fare"}
        with pytest.raises(ValidationError) as exc_info:
            DeriveColumnParams(**payload)

        assert "Field required" in str(exc_info.value)
        assert "expression" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `dtype` field.

        The `extra="forbid"` rule must apply to prevent schema drift.
        """
        payload: dict[str, Any] = {
            "new_column": "total_fare",
            "expression": "fare + taxes",
            "dtype": "float32",
        }
        with pytest.raises(ValidationError) as exc_info:
            DeriveColumnParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestEvaluateMetricsParams:
    """Tests validating the strict schema for standalone metric evaluation."""

    def test_accepts_valid_payload_and_scrubs_numpy_floats(self) -> None:
        """Inject a valid payload containing NumPy floats in the `values` dict.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.float64` values into native Python `float`s.
        """
        payload: dict[str, Any] = {
            "expression": "revenue / cost",
            "values": {"revenue": np.float64(1000.0), "cost": np.float64(200.0)},
        }
        model = EvaluateMetricsParams(**payload)

        assert model.expression == "revenue / cost"
        assert model.values["revenue"] == 1000.0
        assert isinstance(model.values["revenue"], float)
        assert not isinstance(model.values["revenue"], np.floating)

    def test_rejects_empty_expression_string(self) -> None:
        """Inject an empty string for the `expression` field.

        The schema enforces `min_length=1` to prevent empty `numexpr` evaluations.
        """
        payload: dict[str, Any] = {"expression": "", "values": {"a": 1.0}}
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**payload)

        assert "String should have at least 1 character" in str(exc_info.value)

    def test_rejects_empty_values_dictionary(self) -> None:
        """Inject an empty dictionary for the `values` field.

        The schema enforces `min_length=1` because an expression without variables
        is either a constant (which is a no-op) or invalid.
        """
        payload: dict[str, Any] = {"expression": "1 + 1", "values": {}}
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**payload)

        assert "Dictionary should have at least 1 item" in str(exc_info.value)

    def test_rejects_non_float_value_in_values_dict(self) -> None:
        """Inject a list as a value in the `values` dictionary.

        The schema must enforce strict scalar float types to prevent `numexpr`
        from crashing or attempting array operations.
        """
        payload: dict[str, Any] = {
            "expression": "a + b",
            "values": {"a": [1.0, 2.0], "b": 2.0},  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**payload)

        assert "Input should be a valid number" in str(exc_info.value)

    def test_rejects_non_string_key_in_values_dict(self) -> None:
        """Inject a tuple as a key in the `values` dictionary.

        The Hermetic JSON Boundary must reject unhashable/non-string keys before
        Pydantic processes them.
        """
        payload: dict[str, Any] = {
            "expression": "a + b",
            "values": {("a", "b"): 1.0},  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**payload)

        assert "Value of type 'tuple'" in str(exc_info.value)
        assert "has no sanitization rule." in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `precision` field.

        The `extra="forbid"` rule must apply to prevent schema drift.
        """
        payload: dict[str, Any] = {
            "expression": "a + b",
            "values": {"a": 1.0, "b": 2.0},
            "precision": 2,
        }
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

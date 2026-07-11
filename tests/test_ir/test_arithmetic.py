"""Tests for the derive_column and evaluate_metrics IR models."""

import pytest
from pydantic import ValidationError

# Adjust this import path based on exactly what file these live in
from scrygent.ir.arithmetic import DeriveColumnParams, EvaluateMetricsParams


class TestDeriveColumnConstraints:
    """Verifies boundaries for the DeriveColumnParams model."""

    def test_derive_column_valid_instantiation(self):
        """Ensure a valid LLM payload parses correctly."""
        model = DeriveColumnParams(**{
            "new_column": "profit", 
            "expression": "revenue - cost"
        })
        assert model.new_column == "profit"
        assert model.expression == "revenue - cost"

    def test_derive_column_empty_strings_rejected(self):
        """The LLM cannot ask to create an empty column name or provide an empty expression."""
        with pytest.raises(ValidationError) as exc_info:
            DeriveColumnParams(**{"new_column": "", "expression": "revenue - cost"})
        assert "String should have at least 1 character" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info_expr:
            DeriveColumnParams(**{"new_column": "profit", "expression": ""})
        assert "String should have at least 1 character" in str(exc_info_expr.value)


class TestEvaluateMetricsConstraints:
    """Verifies boundaries for the EvaluateMetricsParams model."""

    def test_evaluate_metrics_valid_instantiation(self):
        """Ensure a valid payload with numeric dict values parses correctly."""
        model = EvaluateMetricsParams(**{
            "expression": "a / b",
            "values": {"a": 100.5, "b": 2.0}
        })
        assert model.expression == "a / b"
        assert model.values["a"] == 100.5

    def test_evaluate_metrics_empty_expression_rejected(self):
        """The LLM must provide a mathematical expression."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**{
                "expression": "",
                "values": {"a": 10.0}
            })
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_evaluate_metrics_empty_dict_rejected(self):
        """The LLM cannot submit an empty dictionary for metric evaluation."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**{
                "expression": "a + b",
                "values": {}
            })
        # Pydantic 2.x specifies dictionary length requirements this way
        assert "should have at least 1 item" in str(exc_info.value)

    def test_evaluate_metrics_invalid_value_type_rejected(self):
        """
        The LLM must provide numbers for the values dictionary. 
        Strings that cannot be coerced to float should fail.
        """
        with pytest.raises(ValidationError) as exc_info:
            EvaluateMetricsParams(**{
                "expression": "a + b",
                "values": {"a": 10.0, "b": "not_a_number"}
            })
        # Pydantic will attempt to coerce the string to a float and fail
        assert "Input should be a valid number" in str(exc_info.value)

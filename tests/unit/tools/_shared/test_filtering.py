"""Destructive test suite for the shared row-filtering engine.

This module aggressively tests the deterministic filter grammar. It ensures
that hallucinated columns, invalid operators, type mismatches, and missing
keys are explicitly rejected with actionable error messages to fuel the
self-healing correction loop.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scrygent.tools._shared.filtering import apply_filters


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a DataFrame with mixed types, nulls, and string values for edge cases."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "Alice"],
        "age": [25, 30, np.nan, 22],
        "score": [85.5, 90.0, 75.0, 85.5],
    })


class TestApplyFiltersValidation:
    """Tests validating strict structural and type enforcement of filter payloads."""

    def test_rejects_missing_keys_in_filter_spec(self, sample_df: pd.DataFrame) -> None:
        """Inject a filter dictionary missing the 'value' key.

        The engine must raise a ValueError pinpointing the missing keys rather
        than crashing with a KeyError downstream.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": ">"}]

        with pytest.raises(ValueError, match="Invalid filter specification \\(missing keys\\)") as exc_info:
            apply_filters(sample_df, filters)

        assert "{'column': 'age', 'operator': '>'}" in str(exc_info.value)

    def test_rejects_hallucinated_column_and_provides_difflib_hint(self, sample_df: pd.DataFrame) -> None:
        """Inject a filter for a non-existent column 'nme' (typo of 'name').

        The engine must raise a ValueError and use difflib to suggest the exact
        correct column name to guide the LLM's correction.
        """
        filters: list[dict[str, Any]] = [{"column": "nme", "operator": "==", "value": "Alice"}]

        with pytest.raises(ValueError, match="Filter column 'nme' not found in dataset.") as exc_info:
            apply_filters(sample_df, filters)

        assert "Did you mean exact column name 'name'?" in str(exc_info.value)

    def test_rejects_hallucinated_operator_with_exact_error(self, sample_df: pd.DataFrame) -> None:
        """Inject an unsupported operator string like 'equals'.

        The engine must reject the hallucinated operator and list the valid
        options to constrain the LLM's next attempt.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": "equals", "value": 25}]

        with pytest.raises(ValueError, match="Unsupported filter operator: 'equals'. Choose from:") as exc_info:
            apply_filters(sample_df, filters)

        assert "'=='" in str(exc_info.value)
        assert "'contains'" in str(exc_info.value)

    def test_rejects_none_value_with_unsupported_operator(self, sample_df: pd.DataFrame) -> None:
        """Inject a None value alongside a greater-than operator.

        The engine must prevent TypeErrors in Pandas by rejecting None for
        anything other than '==' or '!='.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": ">", "value": None}]

        with pytest.raises(ValueError, match="Operator '>' with None value is not supported. Use '==' or '!='."):
            apply_filters(sample_df, filters)

    def test_rejects_scalar_value_for_in_operator(self, sample_df: pd.DataFrame) -> None:
        """Inject a string value alongside the 'in' operator.

        The engine must enforce the list type for membership queries to prevent
        silent iteration over string characters.
        """
        filters: list[dict[str, Any]] = [{"column": "name", "operator": "in", "value": "Alice"}]

        with pytest.raises(ValueError, match="Operator 'in' requires a list of values, got str."):
            apply_filters(sample_df, filters)


class TestApplyFiltersExecution:
    """Tests validating the deterministic execution and Pandas mask generation."""

    def test_applies_eq_with_none_value_correctly(self, sample_df: pd.DataFrame) -> None:
        """Inject an equality filter checking for missing data (None).

        Asserts the engine successfully routes this to `isna()` and returns
        the exact expected row.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": "==", "value": None}]
        result = apply_filters(sample_df, filters)

        assert len(result) == 1
        assert result["id"].iloc[0] == 3

    def test_applies_neq_with_none_value_correctly(self, sample_df: pd.DataFrame) -> None:
        """Inject an inequality filter checking for non-missing data.

        Asserts the engine routes this to `notna()` and excludes the null row.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": "!=", "value": None}]
        result = apply_filters(sample_df, filters)

        assert len(result) == 3
        assert 3 not in result["id"].tolist()

    def test_provides_close_match_hint_on_zero_row_eq_match(self, sample_df: pd.DataFrame) -> None:
        """Inject an exact match filter for 'Alic' (typo of 'Alice').

        The engine must detect the 0-row match, use difflib to find close matches,
        and raise a ValueError exposing them to the LLM.
        """
        filters: list[dict[str, Any]] = [{"column": "name", "operator": "==", "value": "Alic"}]

        with pytest.raises(
            ValueError, match="Filter returned 0 rows. No exact match for 'Alic' in column 'name'."
        ) as exc_info:
            apply_filters(sample_df, filters)

        assert "Did you mean one of these exact values: ['Alice']?" in str(exc_info.value)

    def test_handles_contains_on_numeric_column_via_string_coercion(self, sample_df: pd.DataFrame) -> None:
        """Inject a 'contains' filter on a numeric column.

        The engine must safely cast the numeric series to string and evaluate
        the condition without crashing.
        """
        filters: list[dict[str, Any]] = [{"column": "id", "operator": "contains", "value": "2"}]
        result = apply_filters(sample_df, filters)

        assert len(result) == 1
        assert result["id"].iloc[0] == 2

    def test_applies_multiple_filters_and_returns_copy(self, sample_df: pd.DataFrame) -> None:
        """Inject multiple valid filters to test sequential mask application.

        Asserts the result is a distinct DataFrame object and the original
        DataFrame is not mutated in place.
        """
        filters: list[dict[str, Any]] = [
            {"column": "age", "operator": ">", "value": 24},
            {"column": "name", "operator": "==", "value": "Alice"},
        ]
        result = apply_filters(sample_df, filters)

        assert len(result) == 1
        assert result["id"].iloc[0] == 1
        assert result is not sample_df
        # Ensure original df is untouched
        assert len(sample_df) == 4

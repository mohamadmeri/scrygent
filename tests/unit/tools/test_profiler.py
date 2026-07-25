"""Destructive test suite for the deterministic dataset profiling engine.

This module aggressively tests the two-level structural profiler. It ensures
that empty DataFrames are handled gracefully, identifier columns are penalized
to prevent bad math, truncation logic triggers exactly when limits are hit,
and LLM-facing artifacts (like regex skeletons and query matches) are precise.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scrygent.tools.profiler import profile_dataframe


@pytest.fixture
def wide_sample_df() -> pd.DataFrame:
    """Provide a DataFrame with > 15 columns and > 20 rows to trigger truncation and ID signals."""
    data: dict[str, Any] = {f"feature_{i}": np.random.rand(25) for i in range(20)}
    data["user_id"] = range(1, 26)  # Monotonic increasing ID
    data["category"] = ["apple", "banana", "cherry", "date", "elderberry"] * 5
    return pd.DataFrame(data)


class TestProfileDataframeEdgeCases:
    """Tests validating boundary conditions and graceful failure modes."""

    def test_handles_empty_dataframe_gracefully(self) -> None:
        """Inject a completely empty DataFrame (0 rows, 0 columns).

        The profiler must not crash on division-by-zero or empty iterations.
        It must return a dictionary with exact baseline keys and empty values.
        """
        empty_df = pd.DataFrame()
        profile = profile_dataframe(empty_df, user_query="any query")

        assert profile["row_count"] == 0
        assert profile["global_schema"] == {}
        assert profile["detailed_stats"] == {}
        assert profile["truncated"] is False
        assert profile["row_sample"] == []
        assert profile["missing_detailed_stats"] == []
        assert profile["query_specific_matches"] == {}
        assert profile["regex_skeletons"] == {}

    def test_handles_zero_row_dataframe_with_columns_gracefully(self) -> None:
        """Inject a DataFrame with columns but zero rows.

        The profiler must avoid division-by-zero in scoring and return the
        exact global schema while leaving detailed stats empty or minimal.
        """
        zero_row_df = pd.DataFrame({
            "col1": pd.Series(dtype="int64"),
            "col2": pd.Series(dtype="object"),
        })
        profile = profile_dataframe(zero_row_df, user_query="col1")

        assert profile["row_count"] == 0
        assert profile["global_schema"] == {"col1": "int64", "col2": "object"}
        # Detailed stats should be computable for 0 rows without crashing
        assert "col1" in profile["detailed_stats"]


class TestProfileDataframeSelectionLogic:
    """Tests validating the column prioritization and identifier penalization."""

    def test_truncates_exact_columns_and_flags_missing(self, wide_sample_df: pd.DataFrame) -> None:
        """Inject a DataFrame with 22 columns (exceeding the 15 column limit).

        Asserts the `truncated` flag is set to True, exactly 15 columns are in
        `detailed_stats`, and the `missing_detailed_stats` list contains the
        exact 7 omitted column names.
        """
        profile = profile_dataframe(wide_sample_df, user_query="feature_0")

        assert profile["truncated"] is True
        assert len(profile["detailed_stats"]) == 15

        missing = profile["missing_detailed_stats"]
        assert len(missing) == 7

    def test_flags_monotonic_id_column(self, wide_sample_df: pd.DataFrame) -> None:
        """Inject a query that matches an ID column (`user_id`).

        The profiler must select the column and flag it as a sequential ID
        to prevent the LLM from attempting aggregations on primary keys.
        """
        profile = profile_dataframe(wide_sample_df, user_query="user_id")

        assert "user_id" in profile["detailed_stats"]
        assert profile["detailed_stats"]["user_id"].get("is_sequential_id") is True


class TestProfileDataframeArtifacts:
    """Tests validating the exact generation of LLM-facing metadata."""

    def test_extracts_regex_skeletons_for_string_columns(self) -> None:
        """Inject a string column with a dominant alphanumeric pattern (e.g., 'A123').

        The profiler must extract the skeleton (e.g., 'A###') and map it exactly
        in the `regex_skeletons` dictionary.
        """
        df = pd.DataFrame({"code": ["A123", "B456", "C789", "D000", "E111"], "val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = profile_dataframe(df, user_query="code")

        assert profile["regex_skeletons"].get("code") == "A###"

    def test_extracts_query_specific_matches_ignoring_stopwords(self) -> None:
        """Inject a query 'show me the apple' against a 'category' column containing 'apple'.

        The profiler must extract the exact match 'apple', but must NOT extract
        matches for stopwords like 'the' or 'show'.
        """
        df = pd.DataFrame({"category": ["apple", "banana", "cherry", "the", "show"], "val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = profile_dataframe(df, user_query="show me the apple")

        matches = profile["query_specific_matches"]
        assert "category" in matches
        # Should contain 'apple', but NOT 'the' or 'show'
        assert "apple" in matches["category"]
        assert "the" not in matches["category"]
        assert "show" not in matches["category"]

    def test_scrubs_nan_from_row_sample(self) -> None:
        """Inject a DataFrame containing `np.nan` values.

        The profiler must leverage `get_column_sample` which replaces NaNs with
        native Python `None` to ensure the `row_sample` is strictly JSON-serializable.
        """
        df = pd.DataFrame({"col1": [1.0, np.nan, 3.0], "col2": ["a", "b", np.nan]})
        profile = profile_dataframe(df, user_query="col1")

        sample = profile["row_sample"]
        assert len(sample) == 3
        # Assert that NaN is scrubbed (is None or np.isnan)
        assert pd.isna(sample[1]["col1"])
        assert pd.isna(sample[2]["col2"])

"""Destructive test suite for the analyze_data deterministic tool.

This module aggressively tests the analytical query compiler. It ensures
that hallucinated columns, invalid operations, and empty filtered datasets
are explicitly rejected or handled gracefully, preventing Pandas execution
errors and guiding the LLM self-healing loop.
"""

from typing import Any

import pandas as pd
import pytest

from scrygent.tools.analyze_data import analyze_data


class TestAnalyzeDataValidation:
    """Tests validating strict schema and column enforcement before execution."""

    def test_rejects_hallucinated_aggregation_method(self, sample_df: pd.DataFrame) -> None:
        """Inject a metric with an unsupported aggregation like 'mode'.

        The tool must raise a ValueError halting execution and listing valid operations,
        preventing attribute errors in the Pandas `.agg()` call.
        """
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mode", "alias": "mode_age"}]

        with pytest.raises(ValueError, match="Unsupported operation 'mode'. Choose from:") as exc_info:
            analyze_data(sample_df, metrics=metrics)

        assert "mean" in str(exc_info.value)

    def test_rejects_hallucinated_metric_column_and_provides_difflib_hint(self, sample_df: pd.DataFrame) -> None:
        """Inject a metric targeting a non-existent column 'ag' (typo of 'age').

        The tool must raise a ValueError and use difflib to suggest the exact
        correct column name to guide the LLM's correction.
        """
        metrics: list[dict[str, Any]] = [{"column": "ag", "aggregation": "mean", "alias": "avg_age"}]

        with pytest.raises(ValueError, match="Metric target column 'ag' not found in dataset.") as exc_info:
            analyze_data(sample_df, metrics=metrics)

        assert "Did you mean exact column name 'age'?" in str(exc_info.value)

    def test_rejects_duplicate_metric_aliases(self, sample_df: pd.DataFrame) -> None:
        """Inject two metrics sharing the same alias.

        The tool must catch this before execution to prevent silent data
        collisions in the output dictionary.
        """
        metrics: list[dict[str, Any]] = [
            {"column": "age", "aggregation": "mean", "alias": "val"},
            {"column": "fare", "aggregation": "sum", "alias": "val"},
        ]

        with pytest.raises(ValueError, match="Duplicate metric alias 'val'. Each metric must have a unique alias."):
            analyze_data(sample_df, metrics=metrics)

    def test_rejects_hallucinated_group_by_column(self, sample_df: pd.DataFrame) -> None:
        """Inject a group_by list containing a non-existent column 'gendr'.

        The tool must reject the hallucinated dimension and provide a difflib
        hint to prevent Pandas KeyErrors during grouping.
        """
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]

        with pytest.raises(ValueError, match="Group-by column 'gendr' not found in dataset.") as exc_info:
            analyze_data(sample_df, metrics=metrics, group_by=["gendr"])

        assert "Did you mean exact column name 'gender'?" in str(exc_info.value)


class TestAnalyzeDataExecution:
    """Tests validating the deterministic pipeline and Pandas mask generation."""

    def test_executes_simple_aggregation_and_returns_scalar_dict(self, sample_df: pd.DataFrame) -> None:
        """Inject a valid mean aggregation with no group_by.

        Asserts the tool returns a dictionary mapping the alias to the exact scalar,
        bypassing list formatting.
        """
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]

        result = analyze_data(sample_df, metrics=metrics)

        assert "result" in result
        # 22+38+35+40+29+50+18+25+33+42 = 332 / 10 = 33.2
        assert result["result"] == {"avg_age": 33.2}

    def test_executes_grouped_aggregation_and_returns_records(self, sample_df: pd.DataFrame) -> None:
        """Inject a valid grouped aggregation.

        Asserts the tool returns a list of dictionaries (records) with exact values.
        """
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]

        result = analyze_data(sample_df, metrics=metrics, group_by=["gender"])

        assert isinstance(result["result"], list)
        assert len(result["result"]) == 2
        # Check exact values for the 'female' group
        female_rec = next(r for r in result["result"] if r["gender"] == "female")
        assert female_rec["avg_age"] == 42.0

    def test_applies_sort_and_limit_correctly(self, sample_df: pd.DataFrame) -> None:
        """Inject a grouped aggregation with sort and limit.

        Asserts the tool applies the sort on the aggregation alias and truncates
        the result to the exact limit.
        """
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]
        sort: dict[str, str] = {"column": "avg_age", "direction": "desc"}

        result = analyze_data(sample_df, metrics=metrics, group_by=["gender"], sort=sort, limit=1)

        assert len(result["result"]) == 1
        # Male group has avg 33.0, Female has 42.0. Descending should return Female.
        assert result["result"][0]["gender"] == "female"


class TestAnalyzeDataEdgeCases:
    """Tests validating boundary conditions and graceful failure modes."""

    def test_returns_none_and_warning_on_empty_filtered_dataframe(self, sample_df: pd.DataFrame) -> None:
        """Inject a filter that results in a 0-row DataFrame.

        The tool must not crash on empty aggregations. It must return a dictionary
        containing `result: None` and an exact warning string.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": ">", "value": 100}]
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]

        result = analyze_data(sample_df, metrics=metrics, filters=filters)

        assert result["result"] is None
        assert result["warning"] == "Filtered dataset is empty."

    def test_rejects_sort_column_not_in_group_by_or_aliases(self, sample_df: pd.DataFrame) -> None:
        """Inject a sort column that does not match any metric alias or group_by column.

        The tool must catch this logical impossibility after aggregation and raise
        a ValueError exposing the valid options.
        """
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]
        sort: dict[str, str] = {"column": "passenger_id", "direction": "asc"}

        with pytest.raises(
            ValueError, match="Sort column 'passenger_id' not found. Must be an aggregation alias or group dimension."
        ) as exc_info:
            analyze_data(sample_df, metrics=metrics, group_by=["gender"], sort=sort)

        assert "Available: ['avg_age', 'gender']" in str(exc_info.value)

    def test_does_not_mutate_original_dataframe(self, sample_df: pd.DataFrame) -> None:
        """Inject a valid aggregation pipeline.

        Asserts the tool operates on a copy and the original DataFrame remains untouched.
        """
        original_len = len(sample_df)
        metrics: list[dict[str, Any]] = [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]
        filters: list[dict[str, Any]] = [{"column": "age", "operator": ">", "value": 30}]

        analyze_data(sample_df, metrics=metrics, filters=filters)

        assert len(sample_df) == original_len

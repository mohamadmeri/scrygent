"""Destructive test suite for the analyze_data Intermediate Representation.

This module aggressively tests the unified analytical query IR schema. It
ensures that hallucinated aggregations, invalid sort directions, duplicate
aliases, and logically impossible sort targets are rejected before reaching
the deterministic execution engine.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scrygent.contracts.analyze_data import Aggregation
from scrygent.ir.analyze_data import AnalyzeDataParams, Metric, SortCondition


class TestMetric:
    """Tests validating the strict schema and closed vocabulary of metric aggregations."""

    def test_accepts_valid_metric_payload(self) -> None:
        """Verify a baseline valid metric payload passes schema validation."""
        payload: dict[str, Any] = {"column": "fare", "aggregation": "mean", "alias": "avg_fare"}
        model = Metric(**payload)

        assert model.column == "fare"
        assert model.aggregation == Aggregation.MEAN
        assert model.alias == "avg_fare"

    def test_rejects_hallucinated_aggregation_method(self) -> None:
        """Inject an unsupported aggregation string like 'mode'.

        The schema must reject hallucinated methods to prevent attribute errors
        in the Pandas execution engine.
        """
        payload: dict[str, Any] = {"column": "fare", "aggregation": "mode", "alias": "mode_fare"}
        with pytest.raises(ValidationError) as exc_info:
            Metric(**payload)

        assert "Input should be" in str(exc_info.value)
        assert "'mean'" in str(exc_info.value)
        assert "input_value='mode'" in str(exc_info.value)

    def test_rejects_empty_column_or_alias_strings(self) -> None:
        """Inject empty strings for `column` and `alias`.

        The schema enforces `min_length=1` to prevent the LLM from omitting
        essential identifiers while satisfying the type requirement.
        """
        payload: dict[str, Any] = {"column": "", "aggregation": "sum", "alias": ""}
        with pytest.raises(ValidationError) as exc_info:
            Metric(**payload)

        assert "String should have at least 1 character" in str(exc_info.value)


class TestSortCondition:
    """Tests validating the strict schema and closed vocabulary of sort conditions."""

    def test_accepts_valid_sort_condition(self) -> None:
        """Verify a baseline valid sort payload passes schema validation."""
        payload: dict[str, Any] = {"column": "avg_fare", "direction": "desc"}
        model = SortCondition(**payload)

        assert model.column == "avg_fare"
        assert model.direction == "desc"

    def test_rejects_hallucinated_sort_direction(self) -> None:
        """Inject an unsupported direction string like 'ascending'.

        The schema must enforce the exact Literal vocabulary ('asc', 'desc')
        to prevent downstream KeyErrors in the execution engine.
        """
        payload: dict[str, Any] = {"column": "avg_fare", "direction": "ascending"}
        with pytest.raises(ValidationError) as exc_info:
            SortCondition(**payload)

        assert "Input should be 'asc' or 'desc'" in str(exc_info.value)


class TestAnalyzeDataParams:
    """Tests validating the composite execution pipeline IR and custom validators."""

    def test_accepts_valid_complete_pipeline_and_scrubs_numpy_limit(self) -> None:
        """Inject a valid pipeline containing a NumPy integer for the limit.

        Asserts the entire Filter -> Group -> Aggregate -> Sort -> Limit chain
        is accepted and the `np.int64` is scrubbed to a native Python `int`.
        """
        payload: dict[str, Any] = {
            "filters": [{"column": "age", "operator": ">", "value": 20}],
            "group_by": ["pclass"],
            "metrics": [{"column": "fare", "aggregation": "mean", "alias": "avg_fare"}],
            "sort": {"column": "avg_fare", "direction": "desc"},
            "limit": np.int64(10),
        }
        model = AnalyzeDataParams(**payload)

        assert model.group_by == ["pclass"]
        assert isinstance(model.limit, int)
        assert not isinstance(model.limit, np.integer)
        assert model.limit == 10

    def test_rejects_zero_or_negative_limit_value(self) -> None:
        """Inject a limit of 0.

        The schema enforces `ge=1` to prevent the LLM from silently truncating
        the result set to zero rows.
        """
        payload: dict[str, Any] = {"limit": 0}
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeDataParams(**payload)

        assert "Input should be greater than or equal to 1" in str(exc_info.value)

    def test_rejects_duplicate_metric_aliases_with_exact_error(self) -> None:
        """Inject two metrics sharing the same alias.

        The custom model validator must catch this and raise a ValueError
        to prevent silent data collisions in the output record.
        """
        payload: dict[str, Any] = {
            "metrics": [
                {"column": "fare", "aggregation": "mean", "alias": "total_fare"},
                {"column": "fare", "aggregation": "sum", "alias": "total_fare"},
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeDataParams(**payload)

        assert "Duplicate metric alias(es): ['total_fare']" in str(exc_info.value)

    def test_rejects_sort_column_not_in_group_by_or_aliases(self) -> None:
        """Inject a sort column that does not match any metric alias or group_by column.

        The custom model validator must catch this logical impossibility and raise
        a ValueError guiding the LLM to the valid options.
        """
        payload: dict[str, Any] = {
            "group_by": ["pclass"],
            "metrics": [{"column": "fare", "aggregation": "mean", "alias": "avg_fare"}],
            "sort": {"column": "non_existent_column", "direction": "asc"},
        }
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeDataParams(**payload)

        assert "sort.column 'non_existent_column' is not a metric alias or group_by column" in str(exc_info.value)
        assert "Valid options: ['avg_fare', 'pclass']" in str(exc_info.value)

    def test_accepts_raw_data_sort_without_metrics_or_groups(self) -> None:
        """Inject a sort condition on raw data without any metrics or group_by.

        The custom model validator must bypass the alias resolution check and
        allow sorting on any string, as the Python tool will validate it
        against `df.columns` at runtime.
        """
        payload: dict[str, Any] = {
            "filters": [{"column": "age", "operator": ">", "value": 20}],
            "sort": {"column": "raw_age", "direction": "asc"},
        }
        model = AnalyzeDataParams(**payload)

        assert model.sort is not None
        assert model.sort.column == "raw_age"

    def test_rejects_extra_fields_in_top_level_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `malicious_field`.

        The `extra="forbid"` rule must apply to the top-level parameters model.
        """
        payload: dict[str, Any] = {"limit": 10, "malicious_field": "fail"}
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeDataParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

"""Unit tests for the AnalyzeDataParams IR models (Metric, SortCondition, AnalyzeDataParams)."""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from scrygent.contracts.analyze_data import Aggregation
from scrygent.ir.analyze_data import (
    AnalyzeDataParams,
    Metric,
    SortCondition,
)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
class TestMetric:
    adapter = TypeAdapter(Metric)

    def test_valid_metric(self):
        m = self.adapter.validate_python(
            {"column": "sales", "aggregation": Aggregation.SUM, "alias": "total_sales"}
        )
        assert m.column == "sales"
        assert m.aggregation == Aggregation.SUM
        assert m.alias == "total_sales"

    def test_column_empty_string_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.adapter.validate_python(
                {"column": "", "aggregation": "sum", "alias": "total"}
            )
        assert "String should have at least 1 character" in str(exc.value)

    def test_alias_empty_string_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.adapter.validate_python(
                {"column": "sales", "aggregation": "sum", "alias": ""}
            )
        assert "String should have at least 1 character" in str(exc.value)

    def test_missing_aggregation_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"column": "sales", "alias": "total"})

    def test_aggregation_invalid_enum_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python(
                {"column": "sales", "aggregation": "avg", "alias": "total"}
            )

    def test_aggregation_accepts_enum_member(self):
        m = self.adapter.validate_python(
            {"column": "x", "aggregation": Aggregation.MEAN, "alias": "m"}
        )
        assert m.aggregation == Aggregation.MEAN

    def test_aggregation_accepts_valid_string(self):
        m = self.adapter.validate_python(
            {"column": "x", "aggregation": "sum", "alias": "s"}
        )
        assert m.aggregation == Aggregation.SUM

    def test_column_and_alias_allow_spaces(self):
        m = self.adapter.validate_python(
            {"column": " col ", "aggregation": "count", "alias": " my alias "}
        )
        assert m.column == " col "
        assert m.alias == " my alias "


# ---------------------------------------------------------------------------
# SortCondition
# ---------------------------------------------------------------------------
class TestSortCondition:
    adapter = TypeAdapter(SortCondition)

    def test_valid_asc(self):
        s = self.adapter.validate_python({"column": "sales", "direction": "asc"})
        assert s.direction == "asc"

    def test_valid_desc(self):
        s = self.adapter.validate_python({"column": "sales", "direction": "desc"})
        assert s.direction == "desc"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.adapter.validate_python({"column": "sales", "direction": "sideways"})
        assert "Input should be 'asc' or 'desc'" in str(exc.value)

    def test_column_empty_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.adapter.validate_python({"column": "", "direction": "asc"})
        assert "String should have at least 1 character" in str(exc.value)

    def test_missing_direction_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"column": "sales"})


# ---------------------------------------------------------------------------
# AnalyzeDataParams – field constraints
# ---------------------------------------------------------------------------
class TestAnalyzeDataParamsFieldConstraints:
    params_adapter = TypeAdapter(AnalyzeDataParams)

    def test_minimal_valid(self):
        p = self.params_adapter.validate_python(
            {"metrics": [{"column": "sales", "aggregation": "sum", "alias": "total"}]}
        )
        assert len(p.metrics) == 1

    def test_metrics_empty_list_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.params_adapter.validate_python({"metrics": []})
        assert "List should have at least 1 item" in str(exc.value)

    def test_limit_zero_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.params_adapter.validate_python(
                {
                    "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                    "limit": 0,
                }
            )
        assert "greater than or equal to 1" in str(exc.value)

    def test_limit_negative_rejected(self):
        with pytest.raises(ValidationError):
            self.params_adapter.validate_python(
                {
                    "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                    "limit": -5,
                }
            )

    def test_limit_one_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "limit": 1,
            }
        )
        assert p.limit == 1

    def test_limit_none_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "limit": None,
            }
        )
        assert p.limit is None

    # filters field
    def test_filters_none_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "filters": None,
            }
        )
        assert p.filters is None

    def test_filters_empty_list_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "filters": [],
            }
        )
        assert p.filters == []

    def test_filters_valid_list_accepted(self):
        """Pass the raw dictionary to mimic the LLM's JSON output perfectly."""
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "filters": [
                    {"column": "col", "operator": "==", "value": "val"}
                ],
            }
        )
        assert len(p.filters) == 1 # type: ignore
        # Optionally verify the discriminator routed it correctly
        assert p.filters[0].column == "col" # type: ignore
        assert p.filters[0].operator == "==" # type: ignore
        assert p.filters[0].value == "val" # type: ignore

    def test_filters_invalid_dict_rejected(self):
        with pytest.raises(ValidationError):
            self.params_adapter.validate_python(
                {
                    "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                    "filters": [{"column": "col"}],  # missing operator & value
                }
            )

    # group_by field
    def test_group_by_none_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "group_by": None,
            }
        )
        assert p.group_by is None

    def test_group_by_empty_list_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "group_by": [],
            }
        )
        assert p.group_by == []

    def test_group_by_list_with_strings_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "group_by": ["region", "year"],
            }
        )
        assert p.group_by == ["region", "year"]

    def test_group_by_empty_string_allowed_currently(self):
        """Current model does not forbid empty strings in group_by."""
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "group_by": ["", "ok"],
            }
        )
        assert p.group_by == ["", "ok"]  # not rejected


# ---------------------------------------------------------------------------
# AnalyzeDataParams – model_validator logic
# ---------------------------------------------------------------------------
class TestAnalyzeDataParamsValidator:
    params_adapter = TypeAdapter(AnalyzeDataParams)

    def test_duplicate_aliases_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.params_adapter.validate_python(
                {
                    "metrics": [
                        {"column": "sales", "aggregation": "sum", "alias": "total"},
                        {"column": "costs", "aggregation": "sum", "alias": "total"},
                    ]
                }
            )
        assert "Duplicate metric alias(es)" in str(exc.value)
        assert "'total'" in str(exc.value)

    def test_case_different_aliases_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [
                    {"column": "sales", "aggregation": "sum", "alias": "total"},
                    {"column": "costs", "aggregation": "sum", "alias": "TOTAL"},
                ]
            }
        )
        assert len(p.metrics) == 2

    def test_sort_by_metric_alias_allowed(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "sales", "aggregation": "sum", "alias": "total"}],
                "sort": {"column": "total", "direction": "desc"},
            }
        )
        assert p.sort.column == "total" # type: ignore

    def test_sort_by_group_by_column_allowed(self):
        p = self.params_adapter.validate_python(
            {
                "group_by": ["region"],
                "metrics": [{"column": "sales", "aggregation": "sum", "alias": "total"}],
                "sort": {"column": "region", "direction": "asc"},
            }
        )
        assert p.sort.column == "region" # type: ignore

    def test_sort_column_unresolvable_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.params_adapter.validate_python(
                {
                    "group_by": ["region"],
                    "metrics": [{"column": "sales", "aggregation": "sum", "alias": "total"}],
                    "sort": {"column": "unknown_col", "direction": "asc"},
                }
            )
        err = str(exc.value)
        assert "is not a metric alias or group_by column" in err
        assert "'region'" in err
        assert "'total'" in err

    def test_sort_without_group_by_only_aliases_valid(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "cnt"}],
                "sort": {"column": "cnt", "direction": "asc"},
            }
        )
        assert p.sort.column == "cnt" # type: ignore

    def test_sort_without_group_by_invalid_column_rejected(self):
        with pytest.raises(ValidationError):
            self.params_adapter.validate_python(
                {
                    "metrics": [{"column": "x", "aggregation": "count", "alias": "cnt"}],
                    "sort": {"column": "something_else", "direction": "asc"},
                }
            )

    def test_sort_none_accepted(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "x", "aggregation": "count", "alias": "c"}],
                "sort": None,
            }
        )
        assert p.sort is None

    def test_multiple_aliases_unique_success(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [
                    {"column": "a", "aggregation": "sum", "alias": "s"},
                    {"column": "b", "aggregation": "mean", "alias": "m"},
                ]
            }
        )
        assert len(p.metrics) == 2

    def test_duplicate_aliases_among_three_metrics_rejected(self):
        with pytest.raises(ValidationError) as exc:
            self.params_adapter.validate_python(
                {
                    "metrics": [
                        {"column": "a", "aggregation": "sum", "alias": "x"},
                        {"column": "b", "aggregation": "count", "alias": "y"},
                        {"column": "c", "aggregation": "min", "alias": "x"},
                    ]
                }
            )
        assert "Duplicate metric alias(es): ['x']" in str(exc.value)

    def test_model_validator_does_not_modify_data(self):
        p = self.params_adapter.validate_python(
            {
                "metrics": [{"column": "sales", "aggregation": "sum", "alias": "total"}],
                "sort": {"column": "total", "direction": "asc"},
            }
        )
        assert p.metrics[0].alias == "total"
        assert p.sort.column == "total" # type: ignore


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    params_adapter = TypeAdapter(AnalyzeDataParams)

    def test_very_long_strings_accepted(self):
        long_name = "a" * 1000
        p = self.params_adapter.validate_python(
            {"metrics": [{"column": long_name, "aggregation": "count", "alias": long_name}]}
        )
        assert p.metrics[0].column == long_name

    def test_unicode_in_column_and_alias(self):
        p = self.params_adapter.validate_python(
            {"metrics": [{"column": "café", "aggregation": "sum", "alias": "café_total"}]}
        )
        assert p.metrics[0].column == "café"

    def test_metrics_list_accepts_non_dict_items_only_if_valid_models(self):
        with pytest.raises(ValidationError):
            self.params_adapter.validate_python({"metrics": [123]})

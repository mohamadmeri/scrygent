"""Unit tests for the AnalyzeDataParams IR models (Metric, SortCondition, AnalyzeDataParams)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scrygent.contracts.analyze_data import Aggregation
from scrygent.ir.analyze_data import (
    AnalyzeDataParams,
    Metric,
    SortCondition,
)
from scrygent.ir.filtering import FilterCondition 


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
class TestMetric:
    def test_valid_metric(self):
        m = Metric(column="sales", aggregation=Aggregation.SUM, alias="total_sales")
        assert m.column == "sales"
        assert m.aggregation == Aggregation.SUM
        assert m.alias == "total_sales"

    def test_column_empty_string_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Metric(column="", aggregation="sum", alias="total") # type: ignore 
        assert "String should have at least 1 character" in str(exc.value)

    def test_alias_empty_string_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Metric(column="sales", aggregation="sum", alias="") # type: ignore 
        assert "String should have at least 1 character" in str(exc.value)

    def test_missing_aggregation_rejected(self):
        with pytest.raises(ValidationError):
            Metric(column="sales", alias="total")# type: ignore 

    def test_aggregation_invalid_enum_rejected(self):
        with pytest.raises(ValidationError):
            Metric(column="sales", aggregation="avg", alias="total")# type: ignore 

    def test_aggregation_accepts_enum_member(self):
        m = Metric(column="x", aggregation=Aggregation.MEAN, alias="m")
        assert m.aggregation == Aggregation.MEAN

    def test_aggregation_accepts_valid_string(self):
        m = Metric(column="x", aggregation="sum", alias="s")# type: ignore 
        assert m.aggregation == Aggregation.SUM

    def test_column_and_alias_allow_spaces(self):
        m = Metric(column=" col ", aggregation="count", alias=" my alias ")# type: ignore 
        assert m.column == " col "
        assert m.alias == " my alias "


# ---------------------------------------------------------------------------
# SortCondition
# ---------------------------------------------------------------------------
class TestSortCondition:
    def test_valid_asc(self):
        s = SortCondition(column="sales", direction="asc")
        assert s.direction == "asc"

    def test_valid_desc(self):
        s = SortCondition(column="sales", direction="desc")
        assert s.direction == "desc"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SortCondition(column="sales", direction="sideways")# type: ignore 
        assert "Input should be 'asc' or 'desc'" in str(exc.value)

    def test_column_empty_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SortCondition(column="", direction="asc")
        assert "String should have at least 1 character" in str(exc.value)

    def test_missing_direction_rejected(self):
        with pytest.raises(ValidationError):
            SortCondition(column="sales") # type: ignore 


# ---------------------------------------------------------------------------
# AnalyzeDataParams – field constraints
# ---------------------------------------------------------------------------
class TestAnalyzeDataParamsFieldConstraints:
    def test_minimal_valid(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "sales", "aggregation": "sum", "alias": "total"}] # type: ignore 
        )
        assert len(params.metrics) == 1

    def test_metrics_empty_list_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeDataParams(metrics=[])
        assert "List should have at least 1 item" in str(exc.value)

    def test_limit_zero_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeDataParams(
                metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
                limit=0,
            )
        assert "greater than or equal to 1" in str(exc.value)

    def test_limit_negative_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeDataParams(
                metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
                limit=-5,
            )

    def test_limit_one_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            limit=1,
        )
        assert params.limit == 1

    def test_limit_none_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            limit=None,
        )
        assert params.limit is None

    # filters field
    def test_filters_none_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            filters=None,
        )
        assert params.filters is None

    def test_filters_empty_list_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            filters=[],
        )
        assert params.filters == []

    def test_filters_valid_list_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            filters=[
                FilterCondition(column="col", operator="==", value="val") # type: ignore
            ],
        )
        assert len(params.filters) == 1 # type: ignore

    def test_filters_invalid_dict_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeDataParams(
                metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
                filters=[{"column": "col"}],   # type: ignore
            )

    # group_by field
    def test_group_by_none_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            group_by=None,
        )
        assert params.group_by is None

    def test_group_by_empty_list_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore
            group_by=[],
        )
        assert params.group_by == []

    def test_group_by_list_with_strings_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore 
            group_by=["region", "year"],
        )
        assert params.group_by == ["region", "year"]

    def test_group_by_empty_string_allowed_currently(self):
        """Current model does not forbid empty strings in group_by."""
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}], # type: ignore 
            group_by=["", "ok"],
        )
        assert params.group_by == ["", "ok"]  # not rejected


# ---------------------------------------------------------------------------
# AnalyzeDataParams – model_validator logic
# ---------------------------------------------------------------------------
class TestAnalyzeDataParamsValidator:
    def test_duplicate_aliases_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeDataParams(
                metrics=[
                    {"column": "sales", "aggregation": "sum", "alias": "total"}, # type: ignore 
                    {"column": "costs", "aggregation": "sum", "alias": "total"},
                ]
            )
        assert "Duplicate metric alias(es)" in str(exc.value)
        assert "'total'" in str(exc.value)

    def test_case_different_aliases_accepted(self):
        """Current validator is case-sensitive; 'total' and 'TOTAL' are distinct."""
        params = AnalyzeDataParams(
            metrics=[
                {"column": "sales", "aggregation": "sum", "alias": "total"},
                {"column": "costs", "aggregation": "sum", "alias": "TOTAL"},
            ] # type: ignore 
        )
        assert len(params.metrics) == 2  # accepted, though may cause downstream issues

    def test_sort_by_metric_alias_allowed(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "sales", "aggregation": "sum", "alias": "total"}], # type: ignore 
            sort={"column": "total", "direction": "desc"}, # type: ignore 
        ) 
        assert params.sort.column == "total" # type: ignore 

    def test_sort_by_group_by_column_allowed(self):
        params = AnalyzeDataParams(
            group_by=["region"],
            metrics=[{"column": "sales", "aggregation": "sum", "alias": "total"}], # type: ignore 
            sort={"column": "region", "direction": "asc"}, # type: ignore 
        )
        assert params.sort.column == "region" # type: ignore 

    def test_sort_column_unresolvable_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeDataParams(
                group_by=["region"],
                metrics=[{"column": "sales", "aggregation": "sum", "alias": "total"}], # type: ignore 
                sort={"column": "unknown_col", "direction": "asc"}, # type: ignore 
            )
        err = str(exc.value)
        assert "is not a metric alias or group_by column" in err
        assert "'region'" in err
        assert "'total'" in err

    def test_sort_without_group_by_only_aliases_valid(self):
        # group_by is None -> valid targets only metric aliases
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "cnt"}],# type: ignore 
            sort={"column": "cnt", "direction": "asc"},# type: ignore 
        )
        assert params.sort.column == "cnt" # type: ignore 

    def test_sort_without_group_by_invalid_column_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeDataParams(
                metrics=[{"column": "x", "aggregation": "count", "alias": "cnt"}],# type: ignore 
                sort={"column": "something_else", "direction": "asc"},# type: ignore 
            )

    def test_sort_none_accepted(self):
        params = AnalyzeDataParams(
            metrics=[{"column": "x", "aggregation": "count", "alias": "c"}],# type: ignore 
            sort=None,
        )
        assert params.sort is None

    def test_multiple_aliases_unique_success(self):
        params = AnalyzeDataParams(
            metrics=[
                {"column": "a", "aggregation": "sum", "alias": "s"},
                {"column": "b", "aggregation": "mean", "alias": "m"},# type: ignore 
            ]
        )
        assert len(params.metrics) == 2

    def test_duplicate_aliases_among_three_metrics_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeDataParams(
                metrics=[
                    {"column": "a", "aggregation": "sum", "alias": "x"},
                    {"column": "b", "aggregation": "count", "alias": "y"},
                    {"column": "c", "aggregation": "min", "alias": "x"},
                ]# type: ignore 
            )
        assert "Duplicate metric alias(es): ['x']" in str(exc.value)

    def test_model_validator_does_not_modify_data(self):
        """The validator should return the same object with valid data."""
        data = {
            "metrics": [{"column": "sales", "aggregation": "sum", "alias": "total"}],
            "sort": {"column": "total", "direction": "asc"},
        }
        params = AnalyzeDataParams(**data)
        assert params.metrics[0].alias == "total"
        assert params.sort.column == "total" # type: ignore 


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_very_long_strings_accepted(self):
        long_name = "a" * 1000
        params = AnalyzeDataParams(
            metrics=[{"column": long_name, "aggregation": "count", "alias": long_name}] # type: ignore 
        )
        assert params.metrics[0].column == long_name

    def test_unicode_in_column_and_alias(self):
        params = AnalyzeDataParams(
            metrics=[
                {"column": "café", "aggregation": "sum", "alias": "café_total"}
            ] # type: ignore 
        )
        assert params.metrics[0].column == "café"

    def test_metrics_list_accepts_non_dict_items_only_if_valid_models(self):
        with pytest.raises(ValidationError):
            # pass an integer instead of a metric dict
            AnalyzeDataParams(metrics=[123])  # type: ignore

import pandas as pd
import numpy as np
import pytest

from scrygent.tools.analyze_data import analyze_data


# Fixtures
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A representative dataset covering common types and edge cases."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        "age": [25, 30, 35, 30, 25, 40],
        "score": [88.5, 92.0, 75.0, 88.5, 90.0, None],
        "city": ["NY", "LA", "NY", "SF", "LA", "SF"],
        "gender": ["F", "M", "M", "F", "F", None],
        "null_col": [1.0, None, 2.0, None, None, 3.0],
    })


@pytest.fixture
def int_group_df() -> pd.DataFrame:
    """DataFrame with integer group-by column to test key‑string conversion."""
    return pd.DataFrame({
        "class": [1, 1, 2, 2, 3],
        "value": [10, 20, 30, 40, 50],
    })


# Basic scalar aggregations
class TestScalarAggregations:
    def test_scalar_mean(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="mean")
        assert result == {"result": 30.833333333333332}  # raw NumPy float; exact value

    def test_scalar_sum(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum")
        assert result["result"] == 185

    def test_scalar_count(self, sample_df):
        result = analyze_data(sample_df, target_column="score", operation="count")
        # count excludes NaN → 5
        assert result["result"] == 5

    def test_scalar_nunique(self, sample_df):
        result = analyze_data(sample_df, target_column="city", operation="nunique")
        assert result["result"] == 3  # NY, LA, SF

    def test_scalar_min(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="min")
        assert result["result"] == 25

    def test_scalar_max(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="max")
        assert result["result"] == 40

    def test_scalar_std(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="std")
        # Pandas default ddof=1
        assert round(result["result"], 3) == round(np.std([25, 30, 35, 30, 25, 40], ddof=1), 3)

    def test_scalar_var(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="var")
        assert round(result["result"], 3) == round(np.var([25, 30, 35, 30, 25, 40], ddof=1), 3)


# Grouped aggregations
class TestGroupedAggregations:
    def test_grouped_mean_single_group(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="mean", group_by=["city"])
        assert "result" in result
        assert result["result"] == {"NY": 30.0, "LA": 27.5, "SF": 35.0}

    def test_grouped_sum_multiple_groups(self, sample_df):
        result = analyze_data(sample_df, target_column="score", operation="sum",
                              group_by=["city", "gender"])
        # Group combos: ('NY','F'): 88.5, ('LA','F'): 90.0, ('SF','F'): 88.5, ('LA','M'): 92.0, ('NY','M'): 75.0, ('SF',nan): NaN
        # Sum for ('SF', NaN) is NaN → should be None later after sanitization, but raw result is NaN
        res = result["result"]
        assert len(res) == 6
        assert "('NY', 'F')" in res or str(("NY", "F")) in res
        # confirm key is string
        for k in res:
            assert isinstance(k, str)

    def test_groupby_integer_keys_converted_to_string(self, int_group_df):
        result = analyze_data(int_group_df, target_column="value", operation="sum", group_by=["class"])
        out = result["result"]
        for k in out.keys():
            assert isinstance(k, str)
        assert out == {"1": 30, "2": 70, "3": 50}
    
    def test_grouped_nunique(self, sample_df):
        result = analyze_data(
            sample_df, 
            target_column="city", 
            operation="nunique", 
            group_by=["gender"]
        )
        
        # F: NY, SF, LA (3) | M: LA, NY (2) | None: SF (1)
        res = result["result"]
        assert res["F"] == 3
        assert res["M"] == 2
        # Pandas stringifies np.nan as "nan" 
        assert res["nan"] == 1

# Filters (all operators + null handling)
class TestFilters:
    def test_filter_equals(self, sample_df):
        result = analyze_data(sample_df, target_column="name", operation="count",
                              filters=[{"column": "city", "operator": "==", "value": "NY"}])
        assert result["result"] == 2  # Alice, Charlie

    def test_filter_not_equals(self, sample_df):
        result = analyze_data(sample_df, target_column="name", operation="count",
                              filters=[{"column": "city", "operator": "!=", "value": "NY"}])
        assert result["result"] == 4

    def test_filter_greater_than(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum",
                              filters=[{"column": "age", "operator": ">", "value": 30}])
        assert result["result"] == 35 + 40  # 75

    def test_filter_less_than(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="count",
                              filters=[{"column": "age", "operator": "<", "value": 30}])
        assert result["result"] == 2  # Alice (25), Eve (25)

    def test_filter_greater_equal(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="count",
                              filters=[{"column": "age", "operator": ">=", "value": 35}])
        assert result["result"] == 2  # Charlie (35), Frank (40)

    def test_filter_less_equal(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="count",
                              filters=[{"column": "age", "operator": "<=", "value": 30}])
        assert result["result"] == 4

    def test_filter_in_list(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum",
                              filters=[{"column": "city", "operator": "in", "value": ["NY", "SF"]}])
        # Alice(25) NY, Charlie(35) NY, Diana(30) SF, Frank(40) SF => 25+35+30+40 = 130
        assert result["result"] == 130

    def test_filter_contains_string(self, sample_df):
        result = analyze_data(sample_df, target_column="name", operation="count",
                              filters=[{"column": "name", "operator": "contains", "value": "i"}])
        # Alice, Charlie, Diana -> 3
        assert result["result"] == 3

    def test_filter_none_equals(self, sample_df):
        # score has one None -> Frank
        result = analyze_data(sample_df, target_column="name", operation="count",
                              filters=[{"column": "score", "operator": "==", "value": None}])
        assert result["result"] == 1

    def test_filter_none_not_equals(self, sample_df):
        # score not None -> 5
        result = analyze_data(sample_df, target_column="name", operation="count",
                              filters=[{"column": "score", "operator": "!=", "value": None}])
        assert result["result"] == 5

    def test_filter_none_with_invalid_operator_raises(self, sample_df):
        with pytest.raises(ValueError, match="Operator '>' with None value is not supported"):
            analyze_data(sample_df, target_column="age", operation="count",
                         filters=[{"column": "score", "operator": ">", "value": None}])

    def test_multiple_filters_and_logic(self, sample_df):
        # age > 30 AND city == "SF"
        result = analyze_data(sample_df, target_column="name", operation="count",
                              filters=[
                                  {"column": "age", "operator": ">", "value": 30},
                                  {"column": "city", "operator": "==", "value": "SF"}
                              ])
        # Only Frank (40, SF) -> 1
        assert result["result"] == 1

    def test_filter_after_grouping(self, sample_df):
        # Group by city, filter only cities with score > 80 (pre‑filter)
        result = analyze_data(sample_df, target_column="age", operation="mean",
                              filters=[{"column": "score", "operator": ">", "value": 80}],
                              group_by=["city"])
        # After filtering score > 80: Alice(88.5), Bob(92.0), Diana(88.5), Eve(90.0). Frank excluded (None)
        # Group by city: NY: Alice(88.5) age 25, LA: Bob(92.0) age 30 + Eve(90.0) age 25, SF: Diana(88.5) age 30
        # Means: NY: 25, LA: (30+25)/2 = 27.5, SF: 30
        assert result["result"] == {"NY": 25.0, "LA": 27.5, "SF": 30.0}


# Sorting and top‑k
class TestSortingAndTopK:
    def test_sort_ascending(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum",
                              group_by=["city"], sort_order="asc")
        # sums: NY=25+35=60, LA=30+25=55, SF=30+40=70 -> sorted asc: LA(55), NY(60), SF(70)
        assert list(result["result"].keys()) == ["LA", "NY", "SF"]
        assert list(result["result"].values()) == [55, 60, 70]

    def test_sort_descending(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum",
                              group_by=["city"], sort_order="desc")
        # sorted desc: SF(70), NY(60), LA(55)
        assert list(result["result"].keys()) == ["SF", "NY", "LA"]

    def test_top_k(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum",
                              group_by=["city"], sort_order="desc", top_k=2)
        # top 2: SF(70), NY(60)
        assert list(result["result"].keys()) == ["SF", "NY"]
        assert len(result["result"]) == 2

    def test_top_k_without_sort_retains_first_n(self, sample_df):
        # Without sort, Pandas .head(2) just takes first 2 groups encountered (not deterministic for order)
        # We'll just verify we get only 2 entries
        result = analyze_data(sample_df, target_column="age", operation="count",
                              group_by=["city"], top_k=2)
        assert len(result["result"]) == 2

    def test_top_k_all(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="sum",
                              group_by=["city"], sort_order="asc", top_k=10)
        # all 3 cities
        assert len(result["result"]) == 3


# Edge cases & error handling
class TestEdgeCasesAndErrors:
    def test_empty_after_filter_returns_warning(self, sample_df):
        result = analyze_data(sample_df, target_column="age", operation="count",
                              filters=[{"column": "age", "operator": ">", "value": 100}])
        assert result == {"result": None, "warning": "Filtered dataset is empty."}

    def test_invalid_operation_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported operation"):
            analyze_data(sample_df, target_column="age", operation="skew")

    def test_invalid_target_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="Target column 'salary' not found"):
            analyze_data(sample_df, target_column="salary", operation="mean")

    def test_invalid_group_by_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="Group-by column 'region' not found"):
            analyze_data(sample_df, target_column="age", operation="mean", group_by=["region"])

    def test_invalid_filter_missing_keys_raises(self, sample_df):
        bad_filter = {"column": "age", "operator": ">"}  # no value
        with pytest.raises(ValueError, match="Invalid filter specification"):
            analyze_data(sample_df, target_column="age", operation="count",
                         filters=[bad_filter])

    def test_invalid_filter_unsupported_operator_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            analyze_data(sample_df, target_column="age", operation="count",
                         filters=[{"column": "age", "operator": "between", "value": [20, 30]}])

    def test_original_df_not_mutated(self, sample_df):
        original_rows = len(sample_df)
        analyze_data(sample_df, target_column="score", operation="mean",
                     filters=[{"column": "age", "operator": ">", "value": 25}])
        assert len(sample_df) == original_rows

    def test_all_numeric_operations_with_nulls(self, sample_df):
        # score has nulls; mean, sum should ignore them
        result_mean = analyze_data(sample_df, target_column="score", operation="mean")
        expected_mean = (88.5 + 92.0 + 75.0 + 88.5 + 90.0) / 5
        assert abs(result_mean["result"] - expected_mean) < 0.001

        result_sum = analyze_data(sample_df, target_column="score", operation="sum")
        assert result_sum["result"] == 88.5 + 92.0 + 75.0 + 88.5 + 90.0

    def test_invalid_filter_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="Filter column 'fake_column' not found"):
            analyze_data(sample_df, target_column="age", operation="mean",
                         filters=[{"column": "fake_column", "operator": "==", "value": 1}])

    def test_filter_in_requires_list_raises(self, sample_df):
        with pytest.raises(ValueError, match="requires a list of values"):
            analyze_data(sample_df, target_column="age", operation="count",
                         filters=[{"column": "city", "operator": "in", "value": "NY"}])

# Output format checks
class TestOutputFormat:
    def test_result_key_present(self, sample_df):
        res = analyze_data(sample_df, target_column="age", operation="mean")
        assert "result" in res

    def test_grouped_output_dict_keys_are_strings(self, sample_df):
        res = analyze_data(sample_df, target_column="age", operation="sum", group_by=["city"])
        for k in res["result"].keys():
            assert isinstance(k, str)

    def test_scalar_output_is_returned_directly_in_result(self, sample_df):
        res = analyze_data(sample_df, target_column="age", operation="mean")
        # result is a float (scalar) but wrapped in dict with "result"
        assert isinstance(res["result"], (int, float, np.floating))

"""Tests for analyze_data: scalar, grouped, filtered, sorted, limited, and edge cases."""
import pandas as pd
import numpy as np
import pytest
from scrygent.contracts.analyze_data import Aggregation
from scrygent.tools.analyze_data import analyze_data


# ── Fixtures ──
@pytest.fixture
def base_df() -> pd.DataFrame:
    """Core DataFrame with mixed types, nulls, and multiple columns."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        "age": [25, 30, 35, 30, 25, 40],
        "score": [88.5, 92.0, 75.0, 88.5, 90.0, None],
        "city": ["NY", "LA", "NY", "SF", "LA", "SF"],
        "dept": ["Eng", "Eng", "Sales", "Sales", "HR", "HR"],
        "null_val": [1.0, None, 2.0, None, None, 3.0],
    })

@pytest.fixture
def int_group_df() -> pd.DataFrame:
    """DataFrame with integer group-by column (test string conversion)."""
    return pd.DataFrame({
        "class": [1, 1, 2, 2, 3],
        "value": [10, 20, 30, 40, 50],
    })


# ── Helper to call analyze_data concisely ──
def _call(df, metrics, **kwargs):
    return analyze_data(df, metrics, **kwargs)


# ── Scalar Aggregations ──
class TestScalarAggregations:
    def test_mean(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "avg_age"}])
        assert r["result"]["avg_age"] == pytest.approx(30.833333333333332)

    def test_sum(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}])
        assert r["result"]["s"] == 185

    def test_count(self, base_df):
        r = _call(base_df, [{"column": "score", "aggregation": "count", "alias": "cnt"}])
        # count ignores NaN
        assert r["result"]["cnt"] == 5

    def test_nunique(self, base_df):
        r = _call(base_df, [{"column": "city", "aggregation": "nunique", "alias": "n"}])
        assert r["result"]["n"] == 3  # NY, LA, SF

    def test_min(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "min", "alias": "m"}])
        assert r["result"]["m"] == 25

    def test_max(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "max", "alias": "m"}])
        assert r["result"]["m"] == 40

    def test_std(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "std", "alias": "s"}])
        expected = np.std([25, 30, 35, 30, 25, 40], ddof=1)
        assert r["result"]["s"] == pytest.approx(expected)

    def test_var(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "var", "alias": "v"}])
        expected = np.var([25, 30, 35, 30, 25, 40], ddof=1)
        assert r["result"]["v"] == pytest.approx(expected)

    def test_median(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "median", "alias": "med"}])
        assert r["result"]["med"] == 30  # median of [25,25,30,30,35,40] = (30+30)/2

    def test_multiple_metrics(self, base_df):
        r = _call(base_df, [
            {"column": "age", "aggregation": "mean", "alias": "avg"},
            {"column": "age", "aggregation": "count", "alias": "cnt"},
        ])
        assert "avg" in r["result"]
        assert "cnt" in r["result"]
        assert r["result"]["cnt"] == 6


# ── Grouped Aggregations ──
class TestGroupedAggregations:
    def test_single_group(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "avg"}], group_by=["city"])
        # Result is list of dicts: [{'city': 'LA', 'avg': 27.5}, ...]
        assert len(r["result"]) == 3
        cities = {row["city"] for row in r["result"]}
        assert cities == {"NY", "LA", "SF"}
        for row in r["result"]:
            if row["city"] == "NY":
                assert row["avg"] == 30.0
            elif row["city"] == "LA":
                assert row["avg"] == 27.5
            elif row["city"] == "SF":
                assert row["avg"] == 35.0

    def test_multiple_groups(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "count", "alias": "cnt"}],
                  group_by=["city", "dept"])
        # All combinations
        assert len(r["result"]) == 6
        # Check output structure: each record has city, dept, cnt
        row0 = r["result"][0]
        assert "city" in row0 and "dept" in row0 and "cnt" in row0

    def test_groupby_integer_keys(self, int_group_df):
        r = _call(int_group_df, [{"column": "value", "aggregation": "sum", "alias": "s"}],
                  group_by=["class"])
        # class ints become strings in columns after reset_index, but records have int? actually pandas will preserve int column. But our _format_and_sort_results does reset_index() then columns = [str(c) for c in agg_df.columns], so column names become strings, but values remain original types (int). So records should have class as int, s as int.
        classes = {row["class"] for row in r["result"]}
        assert classes == {1, 2, 3}
        for row in r["result"]:
            if row["class"] == 1:
                assert row["s"] == 30
            elif row["class"] == 2:
                assert row["s"] == 70
            elif row["class"] == 3:
                assert row["s"] == 50

    def test_nunique_with_groupby(self, base_df):
        r = _call(base_df, [{"column": "city", "aggregation": "nunique", "alias": "n"}],
                  group_by=["dept"])
        # Dept Eng: cities LA, NY → 2; Sales: NY, SF → 2; HR: LA, SF → 2
        for row in r["result"]:
            assert row["n"] == 2


# ── Filters (all operators + null) ──
class TestFilters:
    def test_equals(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "city", "operator": "==", "value": "NY"}])
        assert r["result"]["c"] == 2  # Alice, Charlie

    def test_not_equals(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "city", "operator": "!=", "value": "NY"}])
        assert r["result"]["c"] == 4

    def test_gt(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}],
                  filters=[{"column": "age", "operator": ">", "value": 30}])
        assert r["result"]["s"] == 75  # Charlie 35 + Frank 40

    def test_lt(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "age", "operator": "<", "value": 30}])
        assert r["result"]["c"] == 2  # Alice 25, Eve 25

    def test_gte(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "age", "operator": ">=", "value": 35}])
        assert r["result"]["c"] == 2  # Charlie 35, Frank 40

    def test_lte(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "age", "operator": "<=", "value": 30}])
        assert r["result"]["c"] == 4

    def test_in_list(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}],
                  filters=[{"column": "city", "operator": "in", "value": ["NY", "SF"]}])
        # NY: Alice 25, Charlie 35; SF: Diana 30, Frank 40 → sum=130
        assert r["result"]["s"] == 130

    def test_not_in_list(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "city", "operator": "not in", "value": ["NY", "SF"]}])
        # Remaining: LA (Bob, Eve) → 2
        assert r["result"]["c"] == 2

    def test_contains(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "name", "operator": "contains", "value": "li"}])
        # Alice, Charlie, Frank? Frank contains 'li'? No, 'Frank' does not. Actually Alice, Charlie → 2. Wait check: 'Alice' contains 'li', 'Charlie' contains 'li'. So 2.
        assert r["result"]["c"] == 2

    def test_startswith(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "name", "operator": "startswith", "value": "A"}])
        assert r["result"]["c"] == 1  # Alice

    def test_endswith(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "name", "operator": "endswith", "value": "e"}])
        # Alice, Charlie, Eve → 3
        assert r["result"]["c"] == 3

    def test_null_equals(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "score", "operator": "==", "value": None}])
        assert r["result"]["c"] == 1  # Frank

    def test_null_not_equals(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "score", "operator": "!=", "value": None}])
        assert r["result"]["c"] == 5

    def test_null_with_gt_raises(self, base_df):
        with pytest.raises(ValueError, match="Operator '>' with None value is not supported"):
            _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "score", "operator": ">", "value": None}])

    def test_multiple_filters(self, base_df):
        r = _call(base_df, [{"column": "name", "aggregation": "count", "alias": "c"}],
                  filters=[
                      {"column": "age", "operator": ">", "value": 30},
                      {"column": "city", "operator": "==", "value": "SF"}
                  ])
        assert r["result"]["c"] == 1  # Frank

    def test_filter_column_not_found(self, base_df):
        with pytest.raises(ValueError, match="Filter column 'unknown' not found"):
            _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "m"}],
                  filters=[{"column": "unknown", "operator": "==", "value": 1}])

    def test_in_requires_list(self, base_df):
        with pytest.raises(ValueError, match="Operator 'in' requires a list"):
            _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "city", "operator": "in", "value": "NY"}])

    def test_not_in_requires_list(self, base_df):
        with pytest.raises(ValueError, match="Operator 'not in' requires a list"):
            _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "city", "operator": "not in", "value": "NY"}])


# ── Sorting and Top‑k ──
class TestSortingAndLimit:
    def test_sort_ascending(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}],
                  group_by=["city"], sort={"column": "s", "direction": "asc"})
        # Expected order: LA 55, NY 60, SF 70
        vals = [row["s"] for row in r["result"]]
        assert vals == [55, 60, 70]

    def test_sort_descending(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}],
                  group_by=["city"], sort={"column": "s", "direction": "desc"})
        vals = [row["s"] for row in r["result"]]
        assert vals == [70, 60, 55]

    def test_limit(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}],
                  group_by=["city"], sort={"column": "s", "direction": "asc"}, limit=2)
        assert len(r["result"]) == 2
        vals = [row["s"] for row in r["result"]]
        assert vals == [55, 60]

    def test_sort_by_alias_not_found_raises(self, base_df):
        with pytest.raises(ValueError, match="Sort column 'nonexistent' not found"):
            _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}],
                  group_by=["city"], sort={"column": "nonexistent", "direction": "asc"})


# ── Edge Cases & Error Handling ──
class TestEdgeCases:
    def test_empty_after_filter(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "age", "operator": ">", "value": 100}])
        assert r == {"result": None, "warning": "Filtered dataset is empty."}

    def test_invalid_aggregation(self, base_df):
        with pytest.raises(ValueError, match="Unsupported operation"):
            _call(base_df, [{"column": "age", "aggregation": "percentile", "alias": "p"}])

    def test_column_not_found(self, base_df):
        with pytest.raises(ValueError, match="Metric target column 'salary' not found"):
            _call(base_df, [{"column": "salary", "aggregation": "mean", "alias": "m"}])

    def test_group_by_column_not_found(self, base_df):
        with pytest.raises(ValueError, match="Group-by column 'region' not found"):
            _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "m"}],
                  group_by=["region"])

    def test_duplicate_alias(self, base_df):
        with pytest.raises(ValueError, match="Duplicate metric alias 'dup'"):
            _call(base_df, [
                {"column": "age", "aggregation": "mean", "alias": "dup"},
                {"column": "score", "aggregation": "sum", "alias": "dup"},
            ])

    def test_original_df_not_mutated(self, base_df):
        original_columns = base_df.columns.tolist()
        original_len = len(base_df)
        _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "m"}],
              filters=[{"column": "age", "operator": ">", "value": 25}])
        assert len(base_df) == original_len
        assert base_df.columns.tolist() == original_columns

    def test_nulls_in_numeric(self, base_df):
        # score has nulls; aggregations ignore them
        r = _call(base_df, [{"column": "score", "aggregation": "mean", "alias": "m"}])
        assert r["result"]["m"] == pytest.approx((88.5+92.0+75.0+88.5+90.0)/5)

    def test_filter_missing_keys(self, base_df):
        bad_filter = {"column": "age", "operator": ">"}  # no value
        with pytest.raises(ValueError, match="Invalid filter specification"):
            _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[bad_filter])

    def test_filter_unsupported_operator(self, base_df):
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            _call(base_df, [{"column": "age", "aggregation": "count", "alias": "c"}],
                  filters=[{"column": "age", "operator": "between", "value": [20,30]}])


# ── Output Format ──
class TestOutputFormat:
    def test_result_key_present(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "m"}])
        assert "result" in r

    def test_scalar_result_is_dict(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "mean", "alias": "m"}])
        assert isinstance(r["result"], dict)

    def test_grouped_result_is_list_of_dicts(self, base_df):
        r = _call(base_df, [{"column": "age", "aggregation": "sum", "alias": "s"}], group_by=["city"])
        assert isinstance(r["result"], list)
        assert all(isinstance(item, dict) for item in r["result"])
        # Keys should include group columns and metric aliases
        assert "city" in r["result"][0]
        assert "s" in r["result"][0]


# ── Contract: Every Aggregation Works ──
class TestAllAggregations:
    @pytest.mark.parametrize("agg", list(Aggregation))
    def test_every_aggregation_executes(self, agg, base_df):
        # Use a numeric column; for nunique we can also test on categorical but numeric works too.
        col = "age" if agg != Aggregation.NUNIQUE else "city"  # nunique works on any
        result = analyze_data(
            base_df,
            metrics=[{"column": col, "aggregation": agg.value, "alias": "out"}],
        )
        assert "result" in result
        # Output should not be None for non-empty df
        assert result["result"] is not None

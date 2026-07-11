"""Tests for _shared/filtering.py – apply_filters."""
import pandas as pd
import pytest
from scrygent.tools._shared.filtering import apply_filters


class TestApplyFilters:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "num": [1, 2, 3, 4, 5],
            "cat": ["a", "b", "c", "d", "e"],
            "mixed": [1.0, None, 3.0, None, 5.0],
        })

    # ── Basic operators ──
    def test_eq(self, df):
        result = apply_filters(df, [{"column": "num", "operator": "==", "value": 3}])
        assert len(result) == 1
        assert result.iloc[0]["num"] == 3

    def test_neq(self, df):
        result = apply_filters(df, [{"column": "num", "operator": "!=", "value": 3}])
        assert len(result) == 4

    def test_gt(self, df):
        result = apply_filters(df, [{"column": "num", "operator": ">", "value": 3}])
        assert len(result) == 2
        assert all(result["num"] > 3)

    def test_gte(self, df):
        result = apply_filters(df, [{"column": "num", "operator": ">=", "value": 3}])
        assert len(result) == 3

    def test_lt(self, df):
        result = apply_filters(df, [{"column": "num", "operator": "<", "value": 3}])
        assert len(result) == 2

    def test_lte(self, df):
        result = apply_filters(df, [{"column": "num", "operator": "<=", "value": 3}])
        assert len(result) == 3

    # ── IN / NOT_IN ──
    def test_in_list(self, df):
        result = apply_filters(df, [{"column": "num", "operator": "in", "value": [2, 4]}])
        assert len(result) == 2
        assert set(result["num"]) == {2, 4}

    def test_not_in_list(self, df):
        result = apply_filters(df, [{"column": "num", "operator": "not in", "value": [2, 4]}])
        assert len(result) == 3
        assert set(result["num"]) == {1, 3, 5}

    def test_in_requires_list(self, df):
        with pytest.raises(ValueError, match="Operator 'in' requires a list"):
            apply_filters(df, [{"column": "num", "operator": "in", "value": 2}])

    def test_not_in_requires_list(self, df):
        with pytest.raises(ValueError, match="Operator 'not in' requires a list"):
            apply_filters(df, [{"column": "num", "operator": "not in", "value": 2}])

    # ── String operators ──
    def test_contains(self, df):
        result = apply_filters(df, [{"column": "cat", "operator": "contains", "value": "b"}])
        assert len(result) == 1
        assert result.iloc[0]["cat"] == "b"

    def test_startswith(self, df):
        result = apply_filters(df, [{"column": "cat", "operator": "startswith", "value": "a"}])
        assert len(result) == 1

    def test_endswith(self, df):
        result = apply_filters(df, [{"column": "cat", "operator": "endswith", "value": "e"}])
        assert len(result) == 1

    # ── Null handling ──
    def test_eq_null(self, df):
        result = apply_filters(df, [{"column": "mixed", "operator": "==", "value": None}])
        assert len(result) == 2
        assert all(pd.isna(result["mixed"]))

    def test_neq_null(self, df):
        result = apply_filters(df, [{"column": "mixed", "operator": "!=", "value": None}])
        assert len(result) == 3
        assert all(~pd.isna(result["mixed"]))

    def test_unsupported_operator_with_null(self, df):
        with pytest.raises(ValueError, match="Operator '>' with None value is not supported"):
            apply_filters(df, [{"column": "mixed", "operator": ">", "value": None}])

    # ── Error cases ──
    def test_missing_keys(self, df):
        with pytest.raises(ValueError, match="Invalid filter specification"):
            apply_filters(df, [{"column": "num", "operator": "=="}])  # no value

    def test_column_not_found(self, df):
        with pytest.raises(ValueError, match="Filter column 'missing' not found"):
            apply_filters(df, [{"column": "missing", "operator": "==", "value": 1}])

    def test_unsupported_operator(self, df):
        with pytest.raises(ValueError, match="Unsupported filter operator: 'between'"):
            apply_filters(df, [{"column": "num", "operator": "between", "value": [1, 5]}])

    # ── Multiple filters ──
    def test_multiple_filters_and_logic(self, df):
        result = apply_filters(df, [
            {"column": "num", "operator": ">", "value": 2},
            {"column": "cat", "operator": "in", "value": ["c", "d", "e"]},
        ])
        assert len(result) == 3
        assert all(result["num"] > 2)
        assert all(result["cat"].isin(["c", "d", "e"]))

    # ── Empty filter list ──
    def test_empty_filters_returns_copy(self, df):
        result = apply_filters(df, [])
        assert len(result) == len(df)
        assert result.equals(df)

    # ── Original DataFrame not modified ──
    def test_original_df_not_mutated(self, df):
        original_len = len(df)
        apply_filters(df, [{"column": "num", "operator": ">", "value": 2}])
        assert len(df) == original_len

    def test_null_filter_then_another_filter(self, df):
        result = apply_filters(df, [
            {"column": "mixed", "operator": "!=", "value": None},
            {"column": "num", "operator": ">", "value": 2},
        ])
        # After excluding nulls (rows 1,3,5), then num>2 leaves rows 3,5 → length 2
        assert len(result) == 2
        assert set(result["num"]) == {3, 5}

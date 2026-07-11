"""Tests for _shared/column_stats.py – compute_detailed_stats."""
import pandas as pd
import pytest
from scrygent.tools._shared.column_stats import compute_detailed_stats


class TestComputeDetailedStats:
    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": []}, dtype="float64")   # 0 rows, numeric column
        result = compute_detailed_stats(df, ["a"])
        assert result["a"]["null_rate"] == 0.0
        assert "min" in result["a"]       # exists because the column is numeric
        assert pd.isna(result["a"]["min"]) # NaN because no data

    def test_numeric_column_basic(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = compute_detailed_stats(df, ["x"])
        stats = result["x"]
        assert stats["dtype"] == "int64"
        assert stats["null_rate"] == 0.0
        assert stats["unique_count"] == 5
        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["mean"] == 3.0

    def test_float_rounding(self):
        df = pd.DataFrame({"x": [1.23456, 2.34567, 3.45678]})
        result = compute_detailed_stats(df, ["x"])
        stats = result["x"]
        # mean should be rounded to 4 decimals
        expected_mean = round((1.23456 + 2.34567 + 3.45678) / 3, 4)
        assert stats["mean"] == expected_mean
        assert stats["min"] == round(1.23456, 4)
        assert stats["max"] == round(3.45678, 4)

    def test_null_values(self):
        df = pd.DataFrame({"x": [1, None, 3, None, 5]})
        result = compute_detailed_stats(df, ["x"])
        stats = result["x"]
        assert stats["null_rate"] == round(2 / 5, 4)
        assert stats["unique_count"] == 3  # 1,3,5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_non_numeric_column(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        result = compute_detailed_stats(df, ["name"])
        stats = result["name"]
        assert isinstance(stats["dtype"], str)
        assert stats["null_rate"] == 0.0
        assert stats["unique_count"] == 3
        assert "min" not in stats
        assert "max" not in stats
        assert "mean" not in stats

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
        result = compute_detailed_stats(df, ["a", "b", "c"])
        assert set(result.keys()) == {"a", "b", "c"}
        # numeric
        assert "min" in result["a"]
        assert "min" in result["c"]
        # non-numeric
        assert "min" not in result["b"]

    def test_column_with_all_nulls(self):
        import numpy as np
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})   # numeric float column
        result = compute_detailed_stats(df, ["x"])
        stats = result["x"]
        assert stats["null_rate"] == 1.0
        assert stats["unique_count"] == 0
        # Numeric column → min/max/mean exist but are NaN
        assert pd.isna(stats["min"])
        assert pd.isna(stats["max"])
        assert pd.isna(stats["mean"])

    def test_column_missing_does_not_exist_raises_keyerror(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError):
            compute_detailed_stats(df, ["b"])

    def test_output_key_is_string(self):
        df = pd.DataFrame({"0": [1, 2]})          # string column name
        result = compute_detailed_stats(df, ["0"])
        assert "0" in result
        assert isinstance(list(result.keys())[0], str)

    def test_identical_shape_as_lazy_fetch(self):
        """The output shape must be identical to what profiler produces (ensuring Planner can't tell paths apart)."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        stats = compute_detailed_stats(df, ["x"])
        required_keys = {"dtype", "null_rate", "unique_count", "min", "max", "mean"}
        assert required_keys.issubset(stats["x"].keys())

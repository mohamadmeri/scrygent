"""Tests for wrangling tools: filter_dataset, reset_dataset, normalize_column."""
from pathlib import Path
import pandas as pd
import pytest

from scrygent.tools.wrangling import (
    filter_dataset,
    reset_dataset,
    normalize_column,
)
from scrygent.contracts.wrangling import NormalizeMethod


# ── Helpers ──
def _make_csv(tmp_path, data: dict, name: str = "data.csv") -> str:
    df = pd.DataFrame(data)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def simple_csv(tmp_path):
    return _make_csv(tmp_path, {
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": ["x", "y", "z", "w", "v"],
    })


@pytest.fixture
def numeric_csv(tmp_path):
    return _make_csv(tmp_path, {
        "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        "const": [5, 5, 5, 5, 5],
        "neg": [-1, -2, -3, -4, -5],
    })


@pytest.fixture
def string_csv(tmp_path):
    return _make_csv(tmp_path, {
        "text": ["  hello ", "WORLD", "  Foo  ", "bar", "BAZ"],
    })


# ── filter_dataset ──
class TestFilterDataset:
    def test_single_filter(self, simple_csv):
        result = filter_dataset(simple_csv, filters=[{"column": "a", "operator": ">", "value": 2}])
        assert result["row_count"] == 3
        assert "warning" in result
        assert result["warning"] is None
        assert Path(result["current_csv_path"]).exists()

    def test_multiple_filters(self, simple_csv):
        result = filter_dataset(simple_csv, filters=[
            {"column": "a", "operator": ">", "value": 1},
            {"column": "c", "operator": "in", "value": ["x", "z"]},
        ])
        assert result["row_count"] == 1

    def test_empty_result_warning(self, simple_csv):
        result = filter_dataset(simple_csv, filters=[{"column": "a", "operator": ">", "value": 100}])
        assert result["row_count"] == 0
        assert result["warning"] == "Filtered dataset is empty. Subsequent steps will operate on zero rows."

    def test_missing_column_raises(self, simple_csv):
        with pytest.raises(ValueError, match="Filter column 'd' not found"):
            filter_dataset(simple_csv, filters=[{"column": "d", "operator": "==", "value": 1}])

    def test_invalid_filter_key_raises(self, simple_csv):
        with pytest.raises(ValueError, match="Invalid filter specification"):
            filter_dataset(simple_csv, filters=[{"column": "a", "operator": ">"}])  # no value

    def test_no_filters_raises(self, simple_csv):
        with pytest.raises(ValueError, match="at least one filter condition"):
            filter_dataset(simple_csv, filters=[])

    def test_output_csv_contains_filtered_data(self, simple_csv):
        result = filter_dataset(simple_csv, filters=[{"column": "a", "operator": ">", "value": 2}])
        df = pd.read_csv(result["current_csv_path"])
        assert len(df) == 3
        assert all(df["a"] > 2)


# ── reset_dataset ──
class TestResetDataset:
    def test_reset_existing_path(self, simple_csv):
        result = reset_dataset(simple_csv)
        assert result["current_csv_path"] == str(Path(simple_csv))

    def test_reset_non_existent_path_raises(self):
        with pytest.raises(FileNotFoundError, match="original_csv_path no longer exists"):
            reset_dataset("/nonexistent/path.csv")


# ── normalize_column ──
class TestNormalizeColumn:
    # numeric methods
    def test_min_max(self, numeric_csv):
        result = normalize_column(numeric_csv, column="val", method="min_max")
        assert result["column"] == "val"
        assert result["before"] is not None
        assert result["after"] is not None
        # check that values are between 0 and 1
        df = pd.read_csv(result["current_csv_path"])
        vals = df["val"]
        assert vals.min() == 0.0
        assert vals.max() == 1.0

    def test_z_score(self, numeric_csv):
        result = normalize_column(numeric_csv, column="val", method="z_score")
        df = pd.read_csv(result["current_csv_path"])
        vals = df["val"]
        # mean should be near 0, std near 1
        assert abs(vals.mean()) < 1e-6
        assert abs(vals.std() - 1.0) < 0.1

    def test_log(self, numeric_csv):
        result = normalize_column(numeric_csv, column="val", method="log")
        df = pd.read_csv(result["current_csv_path"])
        vals = df["val"]
        assert all(vals >= 0)

    def test_min_max_constant_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="all values are identical"):
            normalize_column(numeric_csv, column="const", method="min_max")

    def test_z_score_zero_variance_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="zero or undefined variance"):
            normalize_column(numeric_csv, column="const", method="z_score")

    def test_log_non_positive_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="contains non-positive values"):
            normalize_column(numeric_csv, column="neg", method="log")

    # string methods
    def test_strip(self, string_csv):
        result = normalize_column(string_csv, column="text", method="strip")
        df = pd.read_csv(result["current_csv_path"])
        assert df["text"].iloc[0] == "hello"      # "  hello " stripped
        assert df["text"].iloc[2] == "Foo"        # "  Foo  " stripped

    def test_lowercase(self, string_csv):
        result = normalize_column(string_csv, column="text", method="lowercase")
        df = pd.read_csv(result["current_csv_path"])
        assert df["text"].iloc[1] == "world"

    def test_uppercase(self, string_csv):
        result = normalize_column(string_csv, column="text", method="uppercase")
        df = pd.read_csv(result["current_csv_path"])
        assert df["text"].iloc[0] == "  HELLO "   # uppercase keeps spaces? Actually title? We just do str.upper(), leading spaces remain. "  hello ".upper() -> "  HELLO "
        # that's fine, we only test that string changed to uppercase

    def test_title_case(self, string_csv):
        result = normalize_column(string_csv, column="text", method="title_case")
        df = pd.read_csv(result["current_csv_path"])
        # "  hello " -> "  Hello "?
        assert df["text"].iloc[0] == "  Hello "

    # error cases
    def test_invalid_method_raises(self, simple_csv):
        with pytest.raises(ValueError):  # NormalizeMethod coercion fails
            normalize_column(simple_csv, column="a", method="unknown")

    def test_numeric_method_on_string_column(self, string_csv):
        with pytest.raises(ValueError, match="requires a numeric column"):
            normalize_column(string_csv, column="text", method="min_max")

    def test_string_method_on_numeric_column(self, numeric_csv):
        with pytest.raises(ValueError, match="is a string operation"):
            normalize_column(numeric_csv, column="val", method="strip")

    def test_column_not_found(self, simple_csv):
        with pytest.raises(ValueError, match="Column 'z' not found"):
            normalize_column(simple_csv, column="z", method="min_max")

    # test that method enum coercion is used (already tested in contracts, but here we ensure it's called)
    def test_method_is_normalized_to_enum(self, simple_csv):
        result = normalize_column(simple_csv, column="a", method="min_max")
        assert isinstance(result["method"], NormalizeMethod)

    def test_output_csv_exists(self, simple_csv):
        result = normalize_column(simple_csv, column="a", method="min_max")
        assert Path(result["current_csv_path"]).exists()

    def test_after_stats_for_numeric_method(self, numeric_csv):
        result = normalize_column(numeric_csv, column="val", method="min_max")
        assert result["after"] is not None
        assert "min" in result["after"]
        assert "max" in result["after"]
        assert "mean" in result["after"]

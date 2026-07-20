"""Destructive and functional test suite for the deterministic wrangling engine.

This module aggressively tests the data transformation tools. It ensures
that invalid methods, hallucinated columns, and mathematical edge cases
(like zero variance) are strictly rejected, while also validating the
multi-step composition pattern via secure CSV path swapping.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scrygent.tools.wrangling import filter_dataset, normalize_column, reset_dataset


@pytest.fixture
def edge_case_csv(tmp_path: Path) -> Path:
    """Provide a CSV with edge case columns for normalization tests."""
    df = pd.DataFrame({
        "constant_val": [5, 5, 5, 5],
        "negative_log": [1.0, 2.0, -1.0, 4.0],
        "valid_numeric": [10, 20, 30, 40],
        "string_col": ["A", "B", "C", "D"],
    })
    path = tmp_path / "edge_cases.csv"
    df.to_csv(path, index=False)
    return path


class TestFilterDataset:
    """Tests validating the dataset filtering and CSV swapping mechanics."""

    def test_use_case_executes_valid_filter_and_swaps_path(self, dummy_csv_path: Path) -> None:
        """Inject a valid filter condition against `age`.

        Asserts the tool writes a new CSV to disk, returns the exact row count,
        and does not emit a warning.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": ">", "value": 30}]
        result = filter_dataset(dummy_csv_path, filters=filters)

        assert isinstance(result["current_csv_path"], str)
        new_path = Path(result["current_csv_path"])
        assert new_path.exists()
        assert new_path != dummy_csv_path
        assert result["row_count"] == 2  # 38.0 and 35.0
        assert result["warning"] is None

    def test_rejects_empty_filters_list(self, dummy_csv_path: Path) -> None:
        """Inject an empty list for the `filters` field.

        The tool must explicitly reject no-op filters to prevent wasted cycles.
        """
        with pytest.raises(ValueError, match="filter_dataset requires at least one filter condition."):
            filter_dataset(dummy_csv_path, filters=[])

    def test_returns_exact_warning_on_zero_row_match(self, dummy_csv_path: Path) -> None:
        """Inject a filter condition that matches zero rows.

        The tool must not crash on an empty resulting DataFrame. It must write
        the empty CSV and return the exact warning string.
        """
        filters: list[dict[str, Any]] = [{"column": "age", "operator": ">", "value": 100}]
        result = filter_dataset(dummy_csv_path, filters=filters)

        assert result["row_count"] == 0
        assert result["warning"] == "Filtered dataset is empty. Subsequent steps will operate on zero rows."


class TestResetDataset:
    """Tests validating the dataset reversion tool."""

    def test_use_case_resets_to_original_path_successfully(self, dummy_csv_path: Path) -> None:
        """Inject a valid original CSV path.

        Asserts the tool returns the exact same path string without any mutation.
        """
        result = reset_dataset(dummy_csv_path)

        assert result["current_csv_path"] == str(dummy_csv_path)

    def test_rejects_reset_when_original_path_missing(self, tmp_path: Path) -> None:
        """Inject a path to a file that has been deleted from disk.

        The tool must raise a FileNotFoundError to prevent silent state corruption
        by pointing `current_csv_path` to a ghost file.
        """
        ghost_path = tmp_path / "ghost.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            reset_dataset(ghost_path)

        assert f"Cannot reset: original_csv_path no longer exists at '{ghost_path}'." in str(exc_info.value)


class TestNormalizeColumnValidation:
    """Tests validating the strict schema and type enforcement of normalization."""

    def test_rejects_hallucinated_method(self, dummy_csv_path: Path) -> None:
        """Inject an unsupported method string like 'standardize'.

        The tool must reject the hallucinated method and list valid options.
        """
        with pytest.raises(ValueError, match="Unsupported normalize method 'standardize'. Choose from:") as exc_info:
            normalize_column(dummy_csv_path, column="age", method="standardize")

        assert "'min_max'" in str(exc_info.value)

    def test_rejects_hallucinated_column(self, dummy_csv_path: Path) -> None:
        """Inject a non-existent column 'ghost'.

        The tool must raise a ValueError exposing the exact available columns.
        """
        with pytest.raises(ValueError, match="Column 'ghost' not found.") as exc_info:
            normalize_column(dummy_csv_path, column="ghost", method="min_max")

        assert "Available: ['passenger_id', 'survived', 'age', 'fare', 'embarked']" in str(exc_info.value)

    def test_rejects_string_method_on_numeric_column(self, dummy_csv_path: Path) -> None:
        """Inject a string method ('strip') on a numeric column ('age').

        The tool must enforce dtype alignment to prevent Pandas AttributeError.
        """
        with pytest.raises(ValueError, match="Method 'strip' is a string operation; 'age' has numeric dtype"):
            normalize_column(dummy_csv_path, column="age", method="strip")

    def test_rejects_numeric_method_on_string_column(self, dummy_csv_path: Path) -> None:
        """Inject a numeric method ('z_score') on a string column ('embarked').

        The tool must enforce dtype alignment to prevent Pandas TypeErrors.
        """
        with pytest.raises(ValueError, match="Method 'z_score' requires a numeric column; 'embarked' has dtype"):
            normalize_column(dummy_csv_path, column="embarked", method="z_score")


class TestNormalizeColumnExecution:
    """Tests validating the deterministic execution and mathematical edge cases."""

    def test_use_case_executes_min_max_and_returns_exact_stats(self, edge_case_csv: Path) -> None:
        """Inject a valid request for min_max normalization on `valid_numeric`.

        Asserts the tool computes the exact before/after stats and writes the new CSV.
        """
        result = normalize_column(edge_case_csv, column="valid_numeric", method="min_max")

        assert isinstance(result["current_csv_path"], str)
        assert Path(result["current_csv_path"]).exists()

        assert result["before"] == {"min": 10.0, "max": 40.0, "mean": 25.0}
        # Mean of [0.0, 0.3333, 0.6667, 1.0] is 0.5
        assert result["after"] == {"min": 0.0, "max": 1.0, "mean": 0.5}

    def test_rejects_min_max_on_constant_column(self, edge_case_csv: Path) -> None:
        """Inject a min_max normalization on a column with identical values.

        The tool must catch the zero-division risk and raise a concise ValueError.
        """
        with pytest.raises(
            ValueError, match="Cannot min-max normalize column 'constant_val': all values are identical \\(5\\)."
        ):
            normalize_column(edge_case_csv, column="constant_val", method="min_max")

    def test_rejects_z_score_on_constant_column(self, edge_case_csv: Path) -> None:
        """Inject a z_score normalization on a column with identical values.

        The tool must catch the zero variance and raise a concise ValueError.
        """
        with pytest.raises(
            ValueError, match="Cannot z-score normalize column 'constant_val': zero or undefined variance."
        ):
            normalize_column(edge_case_csv, column="constant_val", method="z_score")

    def test_rejects_log_transform_on_non_positive_values(self, edge_case_csv: Path) -> None:
        """Inject a log transform on a column containing negative values.

        The tool must validate positivity before passing to `np.log` and raise
        a concise ValueError.
        """
        with pytest.raises(
            ValueError, match="Cannot log-transform column 'negative_log': contains non-positive values."
        ):
            normalize_column(edge_case_csv, column="negative_log", method="log")

    def test_use_case_executes_string_uppercase_and_omits_numeric_stats(self, edge_case_csv: Path) -> None:
        """Inject a valid request for uppercase normalization on `string_col`.

        Asserts the tool applies the string method and sets `before`/`after` to None.
        """
        result = normalize_column(edge_case_csv, column="string_col", method="uppercase")

        assert result["before"] is None
        assert result["after"] is None

        # Verify the actual transformation occurred on disk
        transformed_df = pd.read_csv(result["current_csv_path"])
        assert transformed_df["string_col"].iloc[0] == "A"

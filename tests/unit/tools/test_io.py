"""Destructive test suite for core I/O utilities.

This module aggressively tests the disk-boundary functions. It ensures
that missing files, malformed CSVs, and empty DataFrames are handled
strictly, and that NaN values are scrubbed to None to guarantee JSON
compatibility for LLM consumption.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scrygent.tools.io import get_column_sample, load_csv, write_temp_csv


class TestLoadCSV:
    """Tests validating the strict file loading and parsing error handling."""

    def test_loads_valid_csv_and_returns_dataframe(self, dummy_csv_path: Path) -> None:
        """Inject a path to a valid, well-formed CSV.

        Asserts the function returns a Pandas DataFrame with the exact expected
        shape and column names.
        """
        df = load_csv(dummy_csv_path)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 5)
        assert df.columns.tolist() == ["passenger_id", "survived", "age", "fare", "embarked"]

    def test_rejects_nonexistent_file_with_exact_error(self, tmp_path: Path) -> None:
        """Inject a path to a file that does not exist on disk.

        The function must raise a FileNotFoundError with the exact path echoed
        in the message to aid debugging.
        """
        missing_path = tmp_path / "ghost.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_csv(missing_path)

        assert f"Target CSV file does not exist at path: '{missing_path}'" in str(exc_info.value)

    def test_rejects_malformed_csv_with_parser_error(self, tmp_path: Path) -> None:
        """Inject a path to a file containing syntactically invalid CSV data.

        The function must catch Pandas `ParserError` and raise a concise
        `ValueError` preventing the verbose traceback from leaking into the graph.
        """
        malformed_path = tmp_path / "malformed.csv"
        # Write a string with an unterminated quote to force a Pandas parsing error
        malformed_path.write_text('a,b,c\n1,2,"unterminated')

        with pytest.raises(ValueError, match=f"Failed to parse CSV file at '{malformed_path}'."):
            load_csv(malformed_path)

    def test_rejects_empty_file_with_empty_data_error(self, tmp_path: Path) -> None:
        """Inject a path to a completely empty file (0 bytes).

        The function must catch Pandas `EmptyDataError` and raise a concise
        `ValueError` to prevent the graph from crashing on empty uploads.
        """
        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")

        with pytest.raises(ValueError, match=f"Failed to parse CSV file at '{empty_path}'."):
            load_csv(empty_path)


class TestGetColumnSample:
    """Tests validating the extraction of LLM-safe row samples."""

    def test_returns_list_of_dicts_with_exact_keys(self, sample_df: pd.DataFrame) -> None:
        """Inject a valid DataFrame.

        Asserts the function returns a list of dictionaries, extracts exactly
        the top 3 rows by default, and preserves the exact column keys.
        """
        sample = get_column_sample(sample_df)

        assert isinstance(sample, list)
        assert len(sample) == 3
        assert isinstance(sample[0], dict)
        assert set(sample[0].keys()) == set(sample_df.columns)

    def test_scrubs_nan_values_to_none_for_json_safety(self) -> None:
        """Inject a DataFrame containing `np.nan` in string and numeric columns.

        The function must replace NaNs with native Python `None` to prevent
        `json.dumps()` from throwing a `ValueError` during LLM serialization.
        """
        df_with_nans = pd.DataFrame({"col1": [1.0, np.nan, 3.0], "col2": ["a", np.nan, "c"]})

        sample = get_column_sample(df_with_nans, n=3)

        assert sample[0] == {"col1": 1.0, "col2": "a"}
        assert pd.isna(sample[1]["col1"])
        assert pd.isna(sample[1]["col2"])
        assert sample[2] == {"col1": 3.0, "col2": "c"}

    def test_truncates_long_strings_in_sample(self) -> None:
        """Inject a DataFrame containing strings > 200 characters.

        The function must truncate the string and append a suffix to prevent
        prompt window bloat from unstructured text columns.
        """
        long_string = "A" * 250
        df = pd.DataFrame({"text": [long_string]})

        sample = get_column_sample(df, n=1)

        assert len(sample[0]["text"]) < 250
        assert sample[0]["text"].endswith("...[truncated]")

    def test_returns_empty_list_for_empty_dataframe(self) -> None:
        """Inject a completely empty DataFrame (0 rows).

        The function must return an empty list rather than throwing an index error.
        """
        empty_df = pd.DataFrame()

        assert get_column_sample(empty_df) == []


class TestWriteTempCSV:
    """Tests validating the secure writing of transformed DataFrames to disk."""

    def test_writes_valid_dataframe_and_returns_path(self, sample_df: pd.DataFrame) -> None:
        """Inject a valid DataFrame.

        Asserts the function writes the file to disk, returns a valid `Path` object
        ending in `.csv`, and the file content exactly matches the DataFrame.
        """
        path = write_temp_csv(sample_df, prefix="scrygent_test_")

        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".csv"
        assert path.name.startswith("scrygent_test_")

        # Read it back to verify content
        read_back_df = pd.read_csv(path)
        pd.testing.assert_frame_equal(read_back_df, sample_df)

    def test_writes_completely_empty_dataframe_gracefully(self) -> None:
        """Inject a completely empty DataFrame (0 rows, 0 columns).

        The function must not crash when calling `.to_csv()` on an empty DataFrame.
        It should write a 0-byte file and return the path.
        """
        empty_df = pd.DataFrame()

        path = write_temp_csv(empty_df)

        assert path.exists()
        assert path.stat().st_size == 0

    def test_writes_dataframe_with_columns_but_zero_rows(self) -> None:
        """Inject a DataFrame with headers but no data rows.

        The function must write the headers to the file without crashing.
        """
        zero_row_df = pd.DataFrame({"col1": pd.Series(dtype="int64"), "col2": pd.Series(dtype="object")})

        path = write_temp_csv(zero_row_df)

        assert path.exists()
        # File should contain just the headers
        content = path.read_text()
        assert "col1,col2" in content

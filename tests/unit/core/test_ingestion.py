"""Destructive test suite for the data ingestion and sanitization gateway.

This module aggressively tests the pre-flight dataset scrubber. It ensures
that messy column names, duplicate headers, common string nulls, and
whitespace artifacts are strictly normalized into a pristine format before
the deterministic engine ever sees the data.
"""

from pathlib import Path

import pandas as pd
import pytest

from scrygent.core.ingestion import preflight_clean_dataset


@pytest.fixture
def messy_csv_path(tmp_path: Path) -> Path:
    """Provide a CSV with duplicate headers, symbol names, and string nulls."""
    messy_content = "First Name,Age(Years),--Bad--,!!!,First Name\n  Alice  ,25,NULL,X,Bob\nN/A,?,-,Y,NA\n"
    path = tmp_path / "messy.csv"
    path.write_text(messy_content)
    return path


class TestPreflightCleanDataset:
    """Tests validating the strict normalization and sanitization of raw CSV data."""

    def test_normalizes_headers_to_snake_case_and_resolves_collisions(
        self,
        messy_csv_path: Path,
    ) -> None:
        """Inject a CSV with duplicate columns and non-alphanumeric headers.

        Asserts the scrubber converts all headers to valid snake_case, resolves
        the duplicate 'First Name' to 'first_name_1', and renames the pure symbol
        '!!!' header to 'column'.
        """
        clean_path, _ = preflight_clean_dataset(messy_csv_path)
        clean_df = pd.read_csv(clean_path)

        expected_columns = ["first_name", "age_years", "bad", "column", "first_name_1"]
        assert clean_df.columns.tolist() == expected_columns

    def test_generates_exact_physical_to_logical_alias_map(
        self,
        messy_csv_path: Path,
    ) -> None:
        """Inject a CSV with manipulated headers.

        Asserts the returned dictionary exactly maps the new physical column names
        back to their original logical names (including Pandas' auto-mangling) for UI display.
        """
        _, aliases = preflight_clean_dataset(messy_csv_path)

        expected_aliases = {
            "first_name": "First Name",
            "age_years": "Age(Years)",
            "bad": "--Bad--",
            "column": "!!!",
            "first_name_1": "First Name.1",  # Pandas auto-mangles duplicate headers on read
        }
        assert aliases == expected_aliases

    def test_coerces_common_string_nulls_to_true_nans(
        self,
        messy_csv_path: Path,
    ) -> None:
        """Inject a CSV containing 'N/A', 'NULL', '-', '?', and 'NA'.

        Asserts the scrubber converts all of these to true Pandas NaN values
        so the deterministic tools don't try to compute math on strings.
        """
        clean_path, _ = preflight_clean_dataset(messy_csv_path)
        clean_df = pd.read_csv(clean_path)

        # Row 0: 'NULL' in bad
        assert pd.isna(clean_df["bad"].iloc[0])
        # Row 1: 'N/A' in first_name, '?' in age_years, '-' in bad, 'NA' in first_name_1
        assert pd.isna(clean_df["first_name"].iloc[1])
        assert pd.isna(clean_df["age_years"].iloc[1])
        assert pd.isna(clean_df["bad"].iloc[1])
        assert pd.isna(clean_df["first_name_1"].iloc[1])

    def test_strips_whitespace_from_string_cells(
        self,
        messy_csv_path: Path,
    ) -> None:
        """Inject a CSV containing leading/trailing whitespace in string values.

        Asserts the scrubber applies `.strip()` to all object columns to prevent
        silent grouping mismatches downstream.
        """
        clean_path, _ = preflight_clean_dataset(messy_csv_path)
        clean_df = pd.read_csv(clean_path)

        assert clean_df["first_name"].iloc[0] == "Alice"

    def test_writes_clean_dataset_to_new_temp_path_and_returns_path(
        self,
        messy_csv_path: Path,
    ) -> None:
        """Inject a valid messy CSV.

        Asserts the returned object is a valid Path, the file exists on disk,
        and the path differs from the original input path to prevent mutation.
        """
        clean_path, _ = preflight_clean_dataset(messy_csv_path)

        assert isinstance(clean_path, Path)
        assert clean_path.exists()
        assert clean_path != messy_csv_path

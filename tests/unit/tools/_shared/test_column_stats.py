"""Destructive test suite for the shared column statistics engine.

This module aggressively tests the statistical profiling logic. It ensures
that Pandas/NumPy C-types are scrubbed to native Python primitives, empty
or fully null columns do not crash the engine, and logical flags (like
sequential IDs or constant columns) are detected with exact accuracy.
"""

import math

import numpy as np
import pandas as pd
import pytest

from scrygent.tools._shared.column_stats import compute_detailed_stats


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a DataFrame with varied column types for statistical edge cases."""
    return pd.DataFrame({
        "passenger_id": list(range(1, 12)),
        "age": [22.0, 38.0, np.nan, 35.0, 40.0, 29.0, 50.0, 18.0, 25.0, 33.0, 42.0],
        "gender": ["male"] * 10 + ["female"],
        "constant_col": ["N/A"] * 11,
        "all_null": [np.nan] * 11,
    })


class TestComputeDetailedStats:
    """Tests validating the exact statistical shape and boundary scrubbing of the profiler."""

    def test_scrubs_numpy_types_to_native_python_primitives(self, sample_df: pd.DataFrame) -> None:
        """Inject a DataFrame containing NumPy-backed numeric data.

        Asserts that `mean`, `min`, and `max` are returned as native Python floats,
        not `np.float64`, to guarantee JSON serialization safety.
        """
        stats = compute_detailed_stats(sample_df, ["age"])

        assert "age" in stats
        age_stats = stats["age"]

        assert isinstance(age_stats["min"], float)
        assert not isinstance(age_stats["min"], np.floating)
        assert age_stats["min"] == 18.0

        assert isinstance(age_stats["mean"], float)
        assert not isinstance(age_stats["mean"], np.floating)
        # 22+38+35+40+29+50+18+25+33+42 = 332 / 10 = 33.2
        assert age_stats["mean"] == 33.2

    def test_detects_sequential_id_column_with_exact_flag(self, sample_df: pd.DataFrame) -> None:
        """Inject a strictly monotonic integer column with all unique values.

        The engine must set `is_sequential_id: True` to prevent the LLM from
        hallucinating mathematical operations on primary keys.
        """
        stats = compute_detailed_stats(sample_df, ["passenger_id"])

        assert stats["passenger_id"]["is_sequential_id"] is True

    def test_detects_constant_column_with_exact_flag_and_value(self, sample_df: pd.DataFrame) -> None:
        """Inject a column where all values are identical.

        The engine must set `is_constant: True` and extract the exact
        `constant_value` to prevent the LLM from grouping by a useless column.
        """
        stats = compute_detailed_stats(sample_df, ["constant_col"])

        assert stats["constant_col"]["is_constant"] is True
        assert stats["constant_col"]["constant_value"] == "N/A"

    def test_detects_highly_imbalanced_column_with_exact_flag(self, sample_df: pd.DataFrame) -> None:
        """Inject a low-cardinality column heavily skewed towards one value.

        The engine must set `highly_imbalanced: True` and extract the exact
        `dominant_value` to inform the LLM of data skew.
        """
        stats = compute_detailed_stats(sample_df, ["gender"])

        assert stats["gender"]["highly_imbalanced"] is True
        assert stats["gender"]["dominant_value"] == "male"

    def test_handles_100_percent_null_column_without_crashing(self, sample_df: pd.DataFrame) -> None:
        """Inject a column consisting entirely of NaN values.

        The engine must not crash when computing `min`/`max`/`mean` on an
        empty series. It must return `null_rate: 1.0` and `NaN` for math metrics.
        """
        stats = compute_detailed_stats(sample_df, ["all_null"])

        assert stats["all_null"]["null_rate"] == 1.0
        # Pandas returns np.nan for min/max/mean of all-null numeric columns.
        # The Hermetic JSON Boundary will scrub these to None later.
        assert math.isnan(stats["all_null"]["min"])
        assert math.isnan(stats["all_null"]["max"])
        assert math.isnan(stats["all_null"]["mean"])

    def test_handles_empty_dataframe_gracefully(self) -> None:
        """Inject a completely empty DataFrame.

        The engine must avoid division-by-zero on `null_rate` and return
        an empty dictionary.
        """
        empty_df = pd.DataFrame()
        stats = compute_detailed_stats(empty_df, [])
        assert stats == {}

    def test_handles_zero_row_dataframe_with_columns_gracefully(self) -> None:
        """Inject a DataFrame with columns but zero rows.

        The engine must avoid division-by-zero and safely compute null_rate as 0.0.
        """
        zero_row_df = pd.DataFrame({"col1": pd.Series(dtype="int64"), "col2": pd.Series(dtype="object")})
        stats = compute_detailed_stats(zero_row_df, ["col1", "col2"])

        assert stats["col1"]["null_rate"] == 0.0
        assert stats["col2"]["null_rate"] == 0.0

    def test_rejects_hallucinated_columns_with_keyerror(self, sample_df: pd.DataFrame) -> None:
        """Inject a target column list containing a non-existent column name.

        The engine must raise a KeyError rather than silently skipping it,
        forcing the caller to handle the hallucinated column explicitly.
        """
        with pytest.raises(KeyError) as exc_info:
            compute_detailed_stats(sample_df, ["age", "hallucinated_column"])

        assert "'hallucinated_column'" in str(exc_info.value)

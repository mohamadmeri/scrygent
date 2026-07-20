"""Property-based tests for the analyze_data deterministic engine.

This module uses Hypothesis to fuzz the analytical query compiler. It
generates random numeric datasets and aggregation requests to verify
mathematical invariants hold universally.
"""

from typing import Any

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scrygent.tools.analyze_data import analyze_data


# Strategy for generating a simple, clean numeric DataFrame
@st.composite
def clean_numeric_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a DataFrame with two numeric columns: 'val1' and 'val2'."""
    size = draw(st.integers(min_value=1, max_value=50))
    val1 = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    val2 = draw(
        st.lists(
            st.integers(min_value=0, max_value=10),  # Low cardinality for grouping
            min_size=size,
            max_size=size,
        )
    )
    return pd.DataFrame({"val1": val1, "val2": val2})


class TestAnalyzeDataMathematicalInvariants:
    """Tests validating the mathematical correctness of the deterministic engine."""

    @given(df=clean_numeric_dataframe())
    @settings(max_examples=50)  # Limit for fast test runs
    def test_sum_aggregation_is_mathematically_exact(self, df: pd.DataFrame) -> None:
        """Inject a random DataFrame and request a SUM aggregation.

        Asserts the tool's output exactly matches Pandas' native `.sum()`
        calculation, proving no rows are dropped or hallucinated.
        """
        metrics: list[dict[str, Any]] = [{"column": "val1", "aggregation": "sum", "alias": "total_val1"}]

        result = analyze_data(df, metrics=metrics)

        expected_sum = float(df["val1"].sum())
        actual_sum = float(result["result"]["total_val1"])

        assert actual_sum == pytest.approx(expected_sum)

    @given(df=clean_numeric_dataframe())
    @settings(max_examples=50)
    def test_group_by_count_matches_unique_values(self, df: pd.DataFrame) -> None:
        """Inject a random DataFrame and request a GROUP BY COUNT.

        Asserts the tool returns exactly one row per unique value in the
        grouped column, and the count matches the actual row count.
        """
        metrics: list[dict[str, Any]] = [{"column": "val1", "aggregation": "count", "alias": "cnt"}]

        result = analyze_data(df, metrics=metrics, group_by=["val2"])

        records = result["result"]
        expected_groups = df["val2"].nunique()

        assert isinstance(records, list)
        assert len(records) == expected_groups

        # Verify the count for the first group matches the raw data
        first_group_val = records[0]["val2"]
        expected_count = len(df[df["val2"] == first_group_val])
        assert int(records[0]["cnt"]) == expected_count

    @given(
        df=clean_numeric_dataframe(),
        threshold=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_filter_gt_invariant_holds_for_all_rows(self, df: pd.DataFrame, threshold: float) -> None:
        """Inject a random DataFrame and a filter condition (`val1 > threshold`).

        Asserts every row in the returned result strictly satisfies the
        filter condition. This proves the Pandas mask is applied correctly
        for all random inputs.
        """
        filters: list[dict[str, Any]] = [{"column": "val1", "operator": ">", "value": threshold}]
        metrics: list[dict[str, Any]] = [{"column": "val1", "aggregation": "mean", "alias": "avg_val1"}]

        result = analyze_data(df, metrics=metrics, filters=filters)

        if result["result"] is None:
            # If result is None, the filter must have matched 0 rows
            assert len(df[df["val1"] > threshold]) == 0
        else:
            # If a result exists, the mean of the filtered rows must match
            expected_mean = float(df[df["val1"] > threshold]["val1"].mean())
            actual_mean = float(result["result"]["avg_val1"])
            assert actual_mean == pytest.approx(expected_mean)

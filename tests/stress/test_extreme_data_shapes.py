"""Stress tests for extreme data shapes and resource exhaustion.

This module contains tests designed to push the limits of the Profiler
and I/O boundaries. These tests are marked with `@pytest.mark.stress`
and are deselected by default to prevent CI/CD resource bloat.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scrygent.tools.io import write_temp_csv
from scrygent.tools.profiler import MAX_DETAILED_COLUMNS, profile_dataframe

pytestmark = pytest.mark.stress


class TestExtremeDataShapes:
    """Tests validating system stability under extreme data dimensions."""

    def test_wide_matrix_truncates_profiler_without_oom(self) -> None:
        """Generate a 10x10,000 DataFrame to test Profiler truncation limits.

        Asserts the profiler doesn't crash, strictly returns 15 detailed stats,
        and correctly flags the remaining 9,985 columns as missing.
        """
        # 10 rows, 10,000 columns
        df = pd.DataFrame(np.random.rand(10, 10000))
        df.columns = [f"col_{i}" for i in range(10000)]

        profile = profile_dataframe(df, user_query="col_0")

        assert profile["row_count"] == 10
        assert profile["truncated"] is True
        assert len(profile["detailed_stats"]) == MAX_DETAILED_COLUMNS
        assert len(profile["missing_detailed_stats"]) == 10000 - MAX_DETAILED_COLUMNS

    def test_tall_matrix_handles_high_cardinality_and_disk_io(self, tmp_path: Path) -> None:
        """Generate a 500k row DataFrame to test I/O boundaries and value_counts.

        Asserts the Profiler's value_counts doesn't choke on high cardinality
        and write_temp_csv successfully writes the massive file to disk.
        """
        # 500,000 rows, 2 columns
        size = 500_000
        df = pd.DataFrame({
            "numeric": np.random.rand(size),
            "categorical": np.random.choice([f"cat_{i}" for i in range(10000)], size),
        })

        # Test Profiler high cardinality handling
        profile = profile_dataframe(df, user_query="numeric")
        assert profile["row_count"] == size

        # Test I/O boundary
        path = write_temp_csv(df, prefix="scrygent_stress_")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_poison_matrix_regex_skeleton_extraction_on_massive_strings(self) -> None:
        """Generate 1k rows of 50KB strings to test regex skeleton resilience.

        Asserts the _extract_regex_skeleton function doesn't freeze or OOM,
        and correctly truncates massive patterns to protect the LLM prompt window.
        """
        # 1,000 rows, 1 column of 50KB strings
        massive_string = "A" * 50000
        df = pd.DataFrame({"text_blob": [massive_string] * 1000})

        # If this hangs or OOMs, the test will fail via timeout or MemoryError
        profile = profile_dataframe(df, user_query="text_blob")

        # Verify the skeleton was extracted without crashing
        assert "text_blob" in profile["regex_skeletons"]
        skeleton = profile["regex_skeletons"]["text_blob"]

        # Verify it is a string and has been safely truncated
        assert isinstance(skeleton, str)
        assert len(skeleton) <= 115  # 100 chars + len("...[truncated]")
        assert skeleton.endswith("...[truncated]")
        assert skeleton.startswith("A" * 100)

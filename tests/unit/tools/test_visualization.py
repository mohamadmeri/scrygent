"""Destructive test suite for the deterministic visualization engine.

This module aggressively tests the Plotly figure generation and JSON
serialization boundary. It ensures that hallucinated plot types, incorrect
column arities, and missing columns are explicitly rejected, and that
downsampling logic triggers exactly when datasets exceed memory limits.
"""

from pathlib import Path

import pandas as pd
import pytest

from scrygent.tools.visualization import MAX_PLOT_POINTS, generate_plot


class TestGeneratePlotValidation:
    """Tests validating strict schema and column enforcement before execution."""

    def test_rejects_hallucinated_plot_type(self, dummy_csv_path: Path) -> None:
        """Inject an unsupported plot type like 'pie'.

        The tool must reject the hallucinated type and list valid options.
        """
        with pytest.raises(ValueError, match="Unsupported plot type 'pie'. Choose from:") as exc_info:
            generate_plot(dummy_csv_path, plot_type="pie", columns=["age"])

        assert "'bar'" in str(exc_info.value)

    def test_rejects_empty_columns_list(self, dummy_csv_path: Path) -> None:
        """Inject an empty list for the `columns` field.

        The tool must enforce at least one column to prevent empty figure generation.
        """
        with pytest.raises(ValueError, match="generate_plot requires at least 1 column."):
            generate_plot(dummy_csv_path, plot_type="histogram", columns=[])

    def test_rejects_hallucinated_columns_with_exact_error(self, dummy_csv_path: Path) -> None:
        """Inject a list containing a non-existent column 'ghost'.

        The tool must raise a ValueError exposing the exact missing column
        and the available columns list.
        """
        with pytest.raises(ValueError, match="Column\\(s\\) not found: \\['ghost'\\].") as exc_info:
            generate_plot(dummy_csv_path, plot_type="histogram", columns=["ghost"])

        assert "Available: ['passenger_id', 'survived', 'age', 'fare', 'embarked']" in str(exc_info.value)

    def test_rejects_incorrect_arity_for_bar_plot(self, dummy_csv_path: Path) -> None:
        """Inject only 1 column for a bar plot (which requires 2).

        The tool must enforce the exact column count per plot type to prevent
        Pandas KeyError during the groupby operation.
        """
        with pytest.raises(
            ValueError, match="bar plot requires exactly 2 columns: \\[category_column, value_column\\]."
        ):
            generate_plot(dummy_csv_path, plot_type="bar", columns=["age"])


class TestGeneratePlotExecution:
    """Tests validating the deterministic execution and JSON serialization."""

    def test_executes_histogram_and_returns_valid_json_string(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for a histogram on `age`.

        Asserts the tool returns a dictionary containing a JSON string of the
        Plotly figure and a concise description.
        """
        result = generate_plot(dummy_csv_path, plot_type="histogram", columns=["age"])

        assert "plotly_json" in result
        assert isinstance(result["plotly_json"], str)
        assert "description" in result
        assert result["description"] == "Histogram of age"

    def test_executes_scatter_plot_with_webgl_render_mode(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for a scatter plot.

        Asserts the JSON string contains the WebGL render mode to verify
        GPU acceleration is enabled for performance.
        """
        result = generate_plot(dummy_csv_path, plot_type="scatter", columns=["age", "fare"])

        assert "scattergl" in result["plotly_json"]


class TestGeneratePlotDownsampling:
    """Tests validating the memory boundary enforcement via downsampling."""

    def test_triggers_downsampling_on_large_scatter_plot(self, tmp_path: Path) -> None:
        """Inject a DataFrame with > 5000 rows for a scatter plot.

        The tool must downsample to exactly `MAX_PLOT_POINTS` to protect the
        JSON state boundary and append the exact sampling suffix to the description.
        """
        large_df = pd.DataFrame({"x": range(6000), "y": range(6000)})
        large_csv = tmp_path / "large.csv"
        large_df.to_csv(large_csv, index=False)

        result = generate_plot(large_csv, plot_type="scatter", columns=["x", "y"])

        assert f"(Sampled {MAX_PLOT_POINTS} points)" in result["description"]

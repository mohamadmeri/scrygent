"""Destructive test suite for the visualization Intermediate Representation.

This module aggressively tests the Pydantic IR schema for plot generation.
It ensures that hallucinated plot types, invalid column arities, and
malformed payloads are strictly rejected before reaching the execution engine.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from scrygent.contracts.visualization import PlotType
from scrygent.ir.visualization import PlotParams


class TestPlotParams:
    """Tests validating the strict schema and arity constraints of plot IR."""

    def test_accepts_valid_single_column_plot(self) -> None:
        """Verify a baseline valid histogram payload passes schema validation."""
        payload: dict[str, Any] = {"plot_type": PlotType.HISTOGRAM, "columns": ["age"]}
        model = PlotParams(**payload)

        assert model.plot_type == PlotType.HISTOGRAM
        assert model.columns == ["age"]

    def test_accepts_valid_pair_column_plot(self) -> None:
        """Verify a baseline valid scatter plot payload passes schema validation."""
        payload: dict[str, Any] = {"plot_type": PlotType.SCATTER, "columns": ["age", "fare"]}
        model = PlotParams(**payload)

        assert len(model.columns) == 2

    def test_accepts_valid_heatmap_with_multiple_columns(self) -> None:
        """Verify a heatmap payload with more than 2 columns passes validation."""
        payload: dict[str, Any] = {"plot_type": PlotType.HEATMAP, "columns": ["age", "fare", "pclass"]}
        model = PlotParams(**payload)

        assert model.plot_type == PlotType.HEATMAP
        assert len(model.columns) == 3

    def test_rejects_hallucinated_plot_type(self) -> None:
        """Inject an unsupported plot type string like 'pie'.

        The schema must reject hallucinated types to prevent attribute errors
        in the visualization execution engine.
        """
        payload: dict[str, Any] = {"plot_type": "pie", "columns": ["age"]}
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "Input should be" in str(exc_info.value)
        assert "'bar'" in str(exc_info.value)

    def test_rejects_empty_columns_list(self) -> None:
        """Inject an empty list for the `columns` field.

        The schema enforces `min_length=1` to prevent the LLM from emitting
        plot steps without any data.
        """
        payload: dict[str, Any] = {"plot_type": PlotType.HISTOGRAM, "columns": []}
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "List should have at least 1 item" in str(exc_info.value)

    def test_rejects_non_string_elements_in_columns_list(self) -> None:
        """Inject a list containing integers instead of strings.

        The schema must enforce strict string types for column names to prevent
        implicit type coercion bugs in Pandas plotting functions.
        """
        payload: dict[str, Any] = {"plot_type": PlotType.HISTOGRAM, "columns": [123]}  # type: ignore[dict-item]
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "Input should be a valid string" in str(exc_info.value)

    def test_rejects_single_column_arity_violation(self) -> None:
        """Inject two columns for a histogram (which requires exactly 1).

        The custom arity validator must catch this mismatch and raise a
        ValueError preventing the execution engine from crashing.
        """
        payload: dict[str, Any] = {"plot_type": PlotType.HISTOGRAM, "columns": ["age", "fare"]}
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "histogram requires exactly 1 column, got 2." in str(exc_info.value)

    def test_rejects_pair_column_arity_violation(self) -> None:
        """Inject one column for a scatter plot (which requires exactly 2).

        The custom arity validator must catch this mismatch and raise a
        ValueError guiding the LLM to the correct column count.
        """
        payload: dict[str, Any] = {"plot_type": PlotType.SCATTER, "columns": ["age"]}
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "scatter requires exactly 2 columns, got 1." in str(exc_info.value)

    def test_rejects_heatmap_arity_violation(self) -> None:
        """Inject one column for a heatmap (which requires at least 2).

        The custom arity validator must catch this mismatch and prevent
        the execution engine from attempting to compute a correlation matrix
        on a single vector.
        """
        payload: dict[str, Any] = {"plot_type": PlotType.HEATMAP, "columns": ["age"]}
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "heatmap requires at least 2 columns, got 1." in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `color` field.

        The `extra="forbid"` rule must apply to prevent schema drift and
        silent acceptance of unused LLM parameters.
        """
        payload: dict[str, Any] = {
            "plot_type": PlotType.HISTOGRAM,
            "columns": ["age"],
            "color": "red",
        }
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

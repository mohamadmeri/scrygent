"""Tests for the visualization IR models and their arity constraints."""

import pytest
from pydantic import ValidationError

from scrygent.ir.visualization import PlotParams


class TestPlotParamsConstraints:
    """Verifies boundaries and defaults for the PlotParams model."""

    def test_requires_minimum_one_column(self):
        """The LLM cannot request a plot with zero columns."""
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**{
                "plot_type": "histogram",
                "columns": []
            })
        assert "List should have at least 1 item" in str(exc_info.value)

    def test_invalid_plot_type_rejected(self):
        """Ensure hallucinated enum values are caught and rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**{
                "plot_type": "3d_hologram",
                "columns": ["age"]
            })
        assert "Input should be" in str(exc_info.value)

    def test_title_is_optional(self):
        """Ensure title correctly defaults to None or accepts a string."""
        model = PlotParams(**{
            "plot_type": "histogram",
            "columns": ["age"]
        })
        assert model.title is None

        model_with_title = PlotParams(**{
            "plot_type": "histogram",
            "columns": ["age"],
            "title": "Age Distribution"
        })
        assert model_with_title.title == "Age Distribution"


class TestPlotArityLogic:
    """Verifies the cross-field arity logic for different plot types."""

    def test_single_column_plots(self):
        """Histograms and Box plots require exactly 1 column."""
        # Valid instantiations
        PlotParams(**{"plot_type": "histogram", "columns": ["age"]})
        PlotParams(**{"plot_type": "box", "columns": ["salary"]})

        # Invalid (Too many columns)
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**{"plot_type": "histogram", "columns": ["age", "salary"]})
        
        # Note: Enum conversion means self.plot_type will print as PlotType.HISTOGRAM
        assert "requires exactly 1 column, got 2" in str(exc_info.value)

    def test_pair_column_plots(self):
        """Bar, Line, and Scatter plots require exactly 2 columns."""
        # Valid instantiations
        PlotParams(**{"plot_type": "scatter", "columns": ["age", "salary"]})
        PlotParams(**{"plot_type": "bar", "columns": ["department", "count"]})
        PlotParams(**{"plot_type": "line", "columns": ["date", "sales"]})

        # Invalid (Too few columns)
        with pytest.raises(ValidationError) as exc_info_few:
            PlotParams(**{"plot_type": "scatter", "columns": ["age"]})
        assert "requires exactly 2 columns, got 1" in str(exc_info_few.value)

        # Invalid (Too many columns)
        with pytest.raises(ValidationError) as exc_info_many:
            PlotParams(**{"plot_type": "scatter", "columns": ["age", "salary", "bonus"]})
        assert "requires exactly 2 columns, got 3" in str(exc_info_many.value)

    def test_heatmap_plots(self):
        """Heatmaps require at least 2 columns."""
        # Valid (2 columns)
        PlotParams(**{"plot_type": "heatmap", "columns": ["feature_a", "feature_b"]})
        
        # Valid (3+ columns)
        PlotParams(**{"plot_type": "heatmap", "columns": ["a", "b", "c", "d"]})

        # Invalid (Too few columns)
        with pytest.raises(ValidationError) as exc_info:
            PlotParams(**{"plot_type": "heatmap", "columns": ["a"]})
        assert "heatmap requires at least 2 columns, got 1" in str(exc_info.value)

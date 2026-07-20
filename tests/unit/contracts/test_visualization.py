"""Destructive test suite for the visualization tool contracts.

This module ensures the closed-vocabulary PlotType enum strictly enforces its
allowed values and rejects hallucinated or invalid chart identifiers at the
boundary, preventing the executor from dispatching to non-existent plots.
"""

import pytest

from scrygent.contracts.visualization import PlotType


class TestPlotTypeContract:
    """Validates the exact closed vocabulary and type strictness of the plot type enum."""

    def test_plot_type_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected plot types and no others.

        Asserts that the system cannot be extended with new visualizations without
        explicitly modifying this contract, preventing silent routing failures.
        """
        assert len(PlotType) == 6
        assert PlotType.BAR == "bar"
        assert PlotType.LINE == "line"
        assert PlotType.SCATTER == "scatter"
        assert PlotType.HISTOGRAM == "histogram"
        assert PlotType.BOX == "box"
        assert PlotType.HEATMAP == "heatmap"

        members = [member.value for member in PlotType]
        assert set(members) == {"bar", "line", "scatter", "histogram", "box", "heatmap"}

    def test_plot_type_enum_instances_are_strict_strings(self) -> None:
        """Verify that enum members behave strictly as native strings.

        This guarantees seamless JSON serialization when passing the plot type
        from the LLM payload to the deterministic plotting tool.
        """
        assert isinstance(PlotType.BAR, str)
        assert isinstance(PlotType.HEATMAP, str)

    def test_plot_type_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported plot string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates a non-existent visualization like 'pie' or '3d_surface'.
        """
        with pytest.raises(ValueError, match="'pie' is not a valid PlotType"):
            PlotType("pie")

        with pytest.raises(ValueError, match="'3d_surface' is not a valid PlotType"):
            PlotType("3d_surface")

    def test_plot_type_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers or None to prevent implicit type
        coercion bugs in the IR parsing layer.
        """
        with pytest.raises(ValueError):
            PlotType(1)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            PlotType(None)  # type: ignore[arg-type]

    def test_plot_type_enum_rejects_attribute_access_for_unknown_plots(self) -> None:
        """Attempt to access a non-existent plot type via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = PlotType.AREA  # type: ignore[attr-defined]

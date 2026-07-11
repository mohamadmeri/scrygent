"""Enumerations for the visualization tool family."""

from enum import StrEnum


class PlotType(StrEnum):
    """Supported chart types for data visualization."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"

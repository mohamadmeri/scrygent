"""Enums owned by the visualization tool family."""

from enum import StrEnum


class PlotType(StrEnum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"

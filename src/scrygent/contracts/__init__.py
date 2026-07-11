"""Public API for the contracts layer.

This module re-exports all closed-vocabulary enumerations used across the
Scrygent compiler pipeline. It serves as the single import point for
downstream modules requiring strict type boundaries.
"""

from .analyze_data import Aggregation
from .filtering import FilterOperator
from .llm import LLMProvider
from .statistics import CorrelationMethod, OutlierMethod, RegressionMethod
from .tool_names import ToolName
from .visualization import PlotType
from .wrangling import NormalizeMethod

__all__ = [
    "Aggregation",
    "FilterOperator",
    "CorrelationMethod",
    "OutlierMethod",
    "RegressionMethod",
    "PlotType",
    "NormalizeMethod",
    "ToolName",
    "LLMProvider",
]

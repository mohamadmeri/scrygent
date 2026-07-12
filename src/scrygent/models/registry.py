"""Tool-to-IR schema registry.

Provides a single source of truth mapping deterministic tool identifiers
to their strict Pydantic parameter models. Used by the Executor to
validate LLM-generated payloads against the compiler's type boundaries.
"""

from ..base_model import ScrygentBaseModel
from ..contracts import ToolName
from ..ir import (
    AnalyzeDataParams,
    ColumnStatsParams,
    CorrelationParams,
    DeriveColumnParams,
    EvaluateMetricsParams,
    FilterDatasetParams,
    NoParams,
    NormalizeColumnParams,
    OutlierParams,
    PlotParams,
    RegressionParams,
)

TOOL_PARAM_MODELS: dict[ToolName, type[ScrygentBaseModel]] = {
    ToolName.ANALYZE_DATA: AnalyzeDataParams,
    ToolName.FILTER_DATASET: FilterDatasetParams,
    ToolName.NORMALIZE_COLUMN: NormalizeColumnParams,
    ToolName.RESET_DATASET: NoParams,
    ToolName.CORRELATION: CorrelationParams,
    ToolName.REGRESSION: RegressionParams,
    ToolName.DETECT_OUTLIERS: OutlierParams,
    ToolName.REQUEST_COLUMN_STATS: ColumnStatsParams,
    ToolName.GENERATE_PLOT: PlotParams,
    ToolName.DERIVE_COLUMN: DeriveColumnParams,
    ToolName.EVALUATE_METRICS: EvaluateMetricsParams,
}

from ..base_model import ScrygentBaseModel
from ..contracts import ToolName
from ..ir import (
    AnalyzeDataParams,
    FilterDatasetParams,
    NormalizeColumnParams,
    NoParams,
    CorrelationParams,
    RegressionParams,
    OutlierParams,
    ColumnStatsParams,
    PlotParams,
    DeriveColumnParams,
    EvaluateMetricsParams,
)


# --- TOOL PARAM REGISTRY ---
# Single source of truth: tool_name -> strict IR model. This is a shape
# mapping, not procedural logic. The Planner uses this 
# to validate that the LLM's proposed parameters match the 
# tool's expected IR exactly.

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

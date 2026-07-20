"""Destructive test suite for the tool-to-IR schema registry.

This module aggressively tests the central dispatch mapping. It ensures that
every deterministic tool identifier maps to exactly one strict Pydantic
parameter model, and that no hallucinated or unregistered tools can slip
into the executor's dispatch table.
"""

import pytest

from scrygent.base_model import ScrygentBaseModel
from scrygent.contracts.tool_names import ToolName
from scrygent.ir.analyze_data import AnalyzeDataParams
from scrygent.ir.arithmetic import DeriveColumnParams, EvaluateMetricsParams
from scrygent.ir.statistics import ColumnStatsParams, CorrelationParams, OutlierParams, RegressionParams
from scrygent.ir.visualization import PlotParams
from scrygent.ir.wrangling import FilterDatasetParams, NoParams, NormalizeColumnParams
from scrygent.models.registry import TOOL_PARAM_MODELS


class TestToolParamRegistry:
    """Validates the exact closed vocabulary and structural integrity of the tool registry."""

    def test_registry_has_exact_closed_vocabulary(self) -> None:
        """Verify the registry contains exactly the expected 11 tool mappings.

        Asserts that the system cannot be extended with new tool mappings without
        explicitly modifying this contract, preventing silent routing failures.
        """
        assert len(TOOL_PARAM_MODELS) == 11
        assert set(TOOL_PARAM_MODELS.keys()) == set(ToolName)

    def test_registry_maps_tools_to_exact_ir_schemas(self) -> None:
        """Verify each tool maps to its precise Pydantic parameter model.

        Ensures the Executor will validate LLM payloads against the correct strict
        schema, preventing parameter shape mismatches at runtime.
        """
        expected_mapping = {
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

        # Assert exact dictionary equality to catch any unexpected drift
        assert TOOL_PARAM_MODELS == expected_mapping

    def test_all_registered_models_inherit_from_base_model(self) -> None:
        """Verify every registered schema class inherits from `ScrygentBaseModel`.

        Ensures the Hermetic JSON Boundary is universally applied to all tool
        parameters, preventing raw Pandas/NumPy types from leaking into the state.
        """
        for tool_name, model_cls in TOOL_PARAM_MODELS.items():
            assert issubclass(model_cls, ScrygentBaseModel), (
                f"Tool {tool_name} maps to {model_cls.__name__}, which does not inherit from ScrygentBaseModel."
            )

    def test_registry_rejects_hallucinated_tool_access(self) -> None:
        """Attempt to access a non-existent tool mapping via dictionary key.

        Ensures strict fail-fast behavior rather than returning None and causing
        an AttributeError downstream in the Executor.
        """
        with pytest.raises(KeyError):
            _ = TOOL_PARAM_MODELS["train_model"]  # type: ignore[index]

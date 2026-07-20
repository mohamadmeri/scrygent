"""Destructive test suite for the ToolName contract.

This module ensures the closed-vocabulary ToolName enum strictly enforces its
allowed values and rejects hallucinated or invalid tool identifiers at the
boundary, preventing the executor from dispatching to non-existent tools.
"""

import pytest

from scrygent.contracts.tool_names import ToolName


class TestToolNameContract:
    """Validates the exact closed vocabulary and type strictness of the tool name enum."""

    def test_tool_name_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected tools and no others.

        Asserts that the system cannot be extended with new tools without
        explicitly modifying this contract, preventing silent routing failures.
        """
        assert len(ToolName) == 11
        assert ToolName.ANALYZE_DATA == "analyze_data"
        assert ToolName.FILTER_DATASET == "filter_dataset"
        assert ToolName.NORMALIZE_COLUMN == "normalize_column"
        assert ToolName.RESET_DATASET == "reset_dataset"
        assert ToolName.CORRELATION == "correlation"
        assert ToolName.REGRESSION == "regression"
        assert ToolName.DETECT_OUTLIERS == "detect_outliers"
        assert ToolName.REQUEST_COLUMN_STATS == "request_column_stats"
        assert ToolName.GENERATE_PLOT == "generate_plot"
        assert ToolName.DERIVE_COLUMN == "derive_column"
        assert ToolName.EVALUATE_METRICS == "evaluate_metrics"

        members = [member.value for member in ToolName]
        assert set(members) == {
            "analyze_data",
            "filter_dataset",
            "normalize_column",
            "reset_dataset",
            "correlation",
            "regression",
            "detect_outliers",
            "request_column_stats",
            "generate_plot",
            "derive_column",
            "evaluate_metrics",
        }

    def test_tool_name_enum_instances_are_strict_strings(self) -> None:
        """Verify that enum members behave strictly as native strings.

        This guarantees seamless JSON serialization when passing the tool name
        from the LLM payload to the executor dispatcher.
        """
        assert isinstance(ToolName.ANALYZE_DATA, str)
        assert isinstance(ToolName.FILTER_DATASET, str)

    def test_tool_name_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported tool string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates a non-existent tool like 'run_python'.
        """
        with pytest.raises(ValueError, match="'run_python' is not a valid ToolName"):
            ToolName("run_python")

    def test_tool_name_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers or None to prevent implicit type
        coercion bugs in the IR parsing layer.
        """
        with pytest.raises(ValueError):
            ToolName(0)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            ToolName(None)  # type: ignore[arg-type]

    def test_tool_name_enum_rejects_attribute_access_for_unknown_tools(self) -> None:
        """Attempt to access a non-existent tool via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = ToolName.TRAIN_MODEL  # type: ignore[attr-defined]

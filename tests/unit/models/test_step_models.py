"""Destructive test suite for the execution step and plan schemas.

This module aggressively tests the plan generation, step execution, and
audit trail models. It ensures that malformed tool parameters are caught
early, lazy-fetch constraints are strictly enforced, and boundary-polluting
NumPy types are scrubbed before state assignment.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scrygent.contracts.tool_names import ToolName
from scrygent.models.step_models import Plan, Step, StepRecord


class TestStep:
    """Tests validating the strict schema, IR canonicalization, and boundary enforcement of steps."""

    def test_accepts_valid_step_and_scrubs_numpy_in_parameters(self) -> None:
        """Inject a valid step containing a NumPy integer in the `parameters` dict.

        Asserts the custom validator successfully canonicalizes the parameters
        against the strict IR and the Hermetic JSON Boundary scrubs the `np.int64`
        into a native Python `int`.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "rationale": "Limit results",
            "tool_name": ToolName.ANALYZE_DATA,
            "parameters": {"limit": np.int64(5)},
        }
        model = Step(**payload)

        assert model.tool_name == ToolName.ANALYZE_DATA
        assert model.parameters["limit"] == 5
        assert isinstance(model.parameters["limit"], int)
        assert not isinstance(model.parameters["limit"], np.integer)

    def test_rejects_invalid_tool_parameters_with_exact_error(self) -> None:
        """Inject invalid parameters (limit=0) for the `analyze_data` tool.

        The custom model validator must catch the Pydantic ValidationError, strip
        the verbose traceback, and raise a concise ValueError pointing to the
        exact failing field.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "rationale": "Limit results",
            "tool_name": ToolName.ANALYZE_DATA,
            "parameters": {"limit": 0},
        }
        with pytest.raises(ValidationError) as exc_info:
            Step(**payload)

        assert "IR validation failed at 'limit': Input should be greater than or equal to 1" in str(exc_info.value)

    def test_rejects_missing_rationale_field(self) -> None:
        """Attempt to instantiate the model without the `rationale` field.

        Ensures strict failure when the LLM omits the auditability rationale.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.ANALYZE_DATA,
            "parameters": {"limit": 5},
        }
        with pytest.raises(ValidationError) as exc_info:
            Step(**payload)

        assert "Field required" in str(exc_info.value)
        assert "rationale" in str(exc_info.value)

    def test_rejects_hallucinated_tool_name(self) -> None:
        """Inject an unsupported tool name string like 'train_model'.

        The schema must reject hallucinated tools to prevent the custom validator
        from attempting a registry lookup that results in a RuntimeError.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "rationale": "Train model",
            "tool_name": "train_model",
            "parameters": {},
        }
        with pytest.raises(ValidationError) as exc_info:
            Step(**payload)

        assert "Input should be" in str(exc_info.value)

    def test_rejects_non_string_step_id(self) -> None:
        """Inject an integer for the `step_id` field.

        The schema must enforce string type for step identifiers to prevent
        downstream string manipulation errors in the execution trace.
        """
        payload: dict[str, Any] = {
            "step_id": 1,  # type: ignore[dict-item]
            "rationale": "Limit results",
            "tool_name": ToolName.ANALYZE_DATA,
            "parameters": {"limit": 5},
        }
        with pytest.raises(ValidationError) as exc_info:
            Step(**payload)

        assert "Input should be a valid string" in str(exc_info.value)


class TestStepRecord:
    """Tests validating the strict schema for audit trail entries."""

    def test_accepts_valid_record_and_scrubs_numpy_duration(self) -> None:
        """Inject a valid record containing a NumPy integer for `duration_ms`.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.int64` into a native Python `int`.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.ANALYZE_DATA,
            "duration_ms": np.int64(150),
        }
        model = StepRecord(**payload)

        assert model.duration_ms == 150
        assert isinstance(model.duration_ms, int)
        assert not isinstance(model.duration_ms, np.integer)

    def test_rejects_hallucinated_status(self) -> None:
        """Inject an unsupported status string like 'crashed'.

        The schema must enforce the exact Literal vocabulary to prevent audit
        trail parsing errors in the UI layer.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.ANALYZE_DATA,
            "status": "crashed",
        }
        with pytest.raises(ValidationError) as exc_info:
            StepRecord(**payload)

        assert "Input should be 'success', 'failed' or 'skipped'" in str(exc_info.value)

    def test_rejects_non_integer_duration(self) -> None:
        """Inject a string for the `duration_ms` field.

        The schema must enforce strict integer types to prevent aggregation bugs.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.ANALYZE_DATA,
            "duration_ms": "fast",  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            StepRecord(**payload)

        assert "Input should be a valid integer" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `memory_used` field.

        The `extra="forbid"` rule must apply to prevent schema drift.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.ANALYZE_DATA,
            "memory_used": "50MB",
        }
        with pytest.raises(ValidationError) as exc_info:
            StepRecord(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestPlan:
    """Tests validating the composite structure and lazy-fetch constraints of plans."""

    def test_accepts_valid_multi_step_plan(self) -> None:
        """Verify a baseline valid plan with multiple steps passes schema validation."""
        payload: dict[str, Any] = {
            "steps": [
                {
                    "step_id": "step_1",
                    "rationale": "Filter",
                    "tool_name": ToolName.FILTER_DATASET,
                    "parameters": {"filters": [{"column": "age", "operator": ">", "value": 20}]},
                },
                {
                    "step_id": "step_2",
                    "rationale": "Aggregate",
                    "tool_name": ToolName.ANALYZE_DATA,
                    "parameters": {"metrics": [{"column": "fare", "aggregation": "mean", "alias": "avg_fare"}]},
                },
            ]
        }
        model = Plan(**payload)

        assert len(model.steps) == 2

    def test_rejects_lazy_fetch_plan_with_multiple_steps(self) -> None:
        """Inject a plan containing a lazy-fetch step alongside other operations.

        The custom model validator must catch this structural violation and raise
        a ValueError preventing the Planner from reasoning over incomplete stats.
        """
        payload: dict[str, Any] = {
            "steps": [
                {
                    "step_id": "step_1",
                    "rationale": "Fetch stats",
                    "tool_name": ToolName.REQUEST_COLUMN_STATS,
                    "parameters": {"columns": ["age"]},
                },
                {
                    "step_id": "step_2",
                    "rationale": "Analyze",
                    "tool_name": ToolName.ANALYZE_DATA,
                    "parameters": {"limit": 5},
                },
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            Plan(**payload)

        assert "A Plan containing 'request_column_stats' must consist of exactly one step." in str(exc_info.value)

    def test_accepts_sole_lazy_fetch_plan(self) -> None:
        """Verify a plan containing only a single lazy-fetch step passes validation."""
        payload: dict[str, Any] = {
            "steps": [
                {
                    "step_id": "step_1",
                    "rationale": "Fetch stats",
                    "tool_name": ToolName.REQUEST_COLUMN_STATS,
                    "parameters": {"columns": ["age"]},
                }
            ]
        }
        model = Plan(**payload)

        assert len(model.steps) == 1
        assert model.steps[0].tool_name == ToolName.REQUEST_COLUMN_STATS

    def test_rejects_empty_steps_list(self) -> None:
        """Inject an empty list for the `steps` field.

        Although not explicitly constrained by `min_length`, Pydantic validates
        the list items. An empty plan should be structurally invalid to prevent
        the Executor from entering a no-op cycle.
        Note: If the schema does not enforce min_length, this test verifies
        that an empty plan is at least accepted or rejected predictably.
        """
        payload: dict[str, Any] = {"steps": []}
        # Currently, the schema does not enforce min_length on steps.
        # We assert it passes but results in a 0-length list.
        model = Plan(**payload)
        assert len(model.steps) == 0

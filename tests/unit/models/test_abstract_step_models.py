"""Destructive test suite for the Abstract Step models.

This module aggressively tests the pre-optimization planning schemas. It
ensures that hallucinated tool names, missing intent fields, and malformed
plan structures are strictly rejected before the Planner proceeds to the
IR emission phase.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scrygent.contracts.tool_names import ToolName
from scrygent.models.abstract_step_models import AbstractStep, DraftPlan


class TestAbstractStep:
    """Tests validating the strict schema and boundary enforcement of abstract steps."""

    def test_accepts_valid_payload_and_scrubs_numpy_step_id(self) -> None:
        """Inject a valid payload containing a NumPy string for `step_id`.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.str_` into a native Python `str`.
        """
        payload: dict[str, Any] = {
            "step_id": np.str_("step_1"),
            "tool_name": ToolName.FILTER_DATASET,
            "intent_description": "Remove all passengers under the age of 18.",
        }
        model = AbstractStep(**payload)

        assert model.step_id == "step_1"
        assert isinstance(model.step_id, str)
        assert not isinstance(model.step_id, np.str_)
        assert model.tool_name == ToolName.FILTER_DATASET

    def test_rejects_hallucinated_tool_name(self) -> None:
        """Inject an unsupported tool name string like 'train_model'.

        The schema must reject hallucinated tools at this early stage to prevent
        the Optimizer from attempting to map an invalid tool to IR parameters.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": "train_model",
            "intent_description": "Train a random forest.",
        }
        with pytest.raises(ValidationError) as exc_info:
            AbstractStep(**payload)

        assert "Input should be" in str(exc_info.value)
        assert "'filter_dataset'" in str(exc_info.value)

    def test_rejects_missing_intent_description(self) -> None:
        """Attempt to instantiate the model without the `intent_description` field.

        Ensures strict failure when the LLM omits the strategic rationale,
        which is required for the Optimizer to understand the step's purpose.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.FILTER_DATASET,
        }
        with pytest.raises(ValidationError) as exc_info:
            AbstractStep(**payload)

        assert "Field required" in str(exc_info.value)
        assert "intent_description" in str(exc_info.value)

    def test_rejects_non_string_step_id(self) -> None:
        """Inject an integer for the `step_id` field.

        The schema must enforce string type for step identifiers to prevent
        downstream string manipulation errors in the execution trace.
        """
        payload: dict[str, Any] = {
            "step_id": 1,  # type: ignore[dict-item]
            "tool_name": ToolName.FILTER_DATASET,
            "intent_description": "Filter data.",
        }
        with pytest.raises(ValidationError) as exc_info:
            AbstractStep(**payload)

        assert "Input should be a valid string" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `params` field.

        The `extra="forbid"` rule must apply to prevent the LLM from prematurely
        injecting raw parameters before the IR Emitter pass.
        """
        payload: dict[str, Any] = {
            "step_id": "step_1",
            "tool_name": ToolName.FILTER_DATASET,
            "intent_description": "Filter data.",
            "params": {"filters": []},
        }
        with pytest.raises(ValidationError) as exc_info:
            AbstractStep(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestDraftPlan:
    """Tests validating the composite structure and schema enforcement of the draft plan."""

    def test_accepts_valid_plan_with_multiple_steps(self) -> None:
        """Verify a baseline valid multi-step plan passes schema validation."""
        payload: dict[str, Any] = {
            "rationale": "Filter first, then aggregate.",
            "steps": [
                {
                    "step_id": "step_1",
                    "tool_name": ToolName.FILTER_DATASET,
                    "intent_description": "Filter adults.",
                },
                {
                    "step_id": "step_2",
                    "tool_name": ToolName.ANALYZE_DATA,
                    "intent_description": "Calculate mean fare.",
                },
            ],
        }
        model = DraftPlan(**payload)

        assert len(model.steps) == 2
        assert model.rationale == "Filter first, then aggregate."

    def test_rejects_non_list_steps_payload(self) -> None:
        """Inject a dictionary instead of a list for the `steps` field.

        The contract must strictly enforce the list type to prevent iteration
        errors in the Optimizer.
        """
        payload: dict[str, Any] = {
            "rationale": "Filter data.",
            "steps": {"step_id": "step_1", "tool_name": "filter_dataset", "intent_description": "Filter."},
        }
        with pytest.raises(ValidationError) as exc_info:
            DraftPlan(**payload)

        assert "Input should be a valid list" in str(exc_info.value)

    def test_rejects_malformed_step_in_list(self) -> None:
        """Inject a list containing a step dictionary missing `tool_name`.

        The nested Pydantic validation must catch the missing required field
        inside the list item and halt execution.
        """
        payload: dict[str, Any] = {
            "rationale": "Filter data.",
            "steps": [{"step_id": "step_1", "intent_description": "Filter adults."}],
        }
        with pytest.raises(ValidationError) as exc_info:
            DraftPlan(**payload)

        assert "Field required" in str(exc_info.value)
        assert "tool_name" in str(exc_info.value)

    def test_rejects_missing_rationale_field(self) -> None:
        """Attempt to instantiate the model without the global `rationale`.

        Ensures strict failure when the LLM omits the global strategy, which
        is critical for evaluating the coherence of the plan.
        """
        payload: dict[str, Any] = {
            "steps": [
                {
                    "step_id": "step_1",
                    "tool_name": ToolName.FILTER_DATASET,
                    "intent_description": "Filter adults.",
                }
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            DraftPlan(**payload)

        assert "Field required" in str(exc_info.value)
        assert "rationale" in str(exc_info.value)

    def test_rejects_extra_fields_in_top_level_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `optimized` flag.

        The `extra="forbid"` rule must apply to prevent schema drift on the
        top-level plan model.
        """
        payload: dict[str, Any] = {
            "rationale": "Filter data.",
            "steps": [],
            "optimized": True,
        }
        with pytest.raises(ValidationError) as exc_info:
            DraftPlan(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

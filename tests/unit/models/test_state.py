"""Destructive test suite for the central AgentState container.

This module aggressively tests the global state schema. It ensures that
missing critical paths, hallucinated execution statuses, and boundary-
polluting NumPy types are strictly rejected to prevent state corruption
and LangGraph routing failures.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scrygent.models.state import AgentState


class TestAgentState:
    """Tests validating the strict schema and boundary enforcement of the execution state."""

    def test_accepts_valid_minimal_payload_and_sets_defaults(self) -> None:
        """Verify a baseline valid payload passes schema validation and defaults are set.

        Ensures the entry-node invariants (pending status, replan guard false)
        are established correctly when the state is initialized.
        """
        payload: dict[str, Any] = {
            "original_csv_path": "/tmp/data.csv",
            "current_csv_path": "/tmp/data.csv",
            "user_query": "What is the average age?",
        }
        model = AgentState(**payload)

        assert model.eval_mode is False
        assert model.has_replanned is False
        assert model.execution_status == "pending"
        assert model.current_step_index == 0
        assert model.step_outputs == {}

    def test_rejects_missing_original_csv_path(self) -> None:
        """Attempt to instantiate the model without the `original_csv_path` field.

        Ensures strict failure when the immutable baseline data source is omitted.
        """
        payload: dict[str, Any] = {
            "current_csv_path": "/tmp/data.csv",
            "user_query": "Query",
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentState(**payload)

        assert "Field required" in str(exc_info.value)
        assert "original_csv_path" in str(exc_info.value)

    def test_rejects_hallucinated_execution_status(self) -> None:
        """Inject an unsupported status string like 'failed'.

        The schema must enforce the exact Literal vocabulary to prevent the
        LangGraph conditional routing from hitting a KeyError or dead edge.
        """
        payload: dict[str, Any] = {
            "original_csv_path": "/tmp/data.csv",
            "current_csv_path": "/tmp/data.csv",
            "user_query": "Query",
            "execution_status": "failed",
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentState(**payload)

        assert "Input should be 'pending', 'running', 'replan', 'aborted' or 'complete'" in str(exc_info.value)

    def test_scrubs_numpy_int_from_current_step_index(self) -> None:
        """Inject a NumPy integer for `current_step_index`.

        The Hermetic JSON Boundary must intercept and scrub the `np.int64`
        to a native Python `int` before Pydantic freezes the model.
        """
        payload: dict[str, Any] = {
            "original_csv_path": "/tmp/data.csv",
            "current_csv_path": "/tmp/data.csv",
            "user_query": "Query",
            "current_step_index": np.int64(5),
        }
        model = AgentState(**payload)

        assert model.current_step_index == 5
        assert isinstance(model.current_step_index, int)
        assert not isinstance(model.current_step_index, np.integer)

    def test_rejects_leaked_dataframe_in_step_outputs(self) -> None:
        """Inject a raw `pd.DataFrame` into the `step_outputs` dictionary.

        The Hermetic JSON Boundary must explicitly reject DataFrames to enforce
        the stateless-tool contract and prevent memory bloat in the LangGraph state.
        """
        df = pd.DataFrame({"x": [1, 2]})
        payload: dict[str, Any] = {
            "original_csv_path": "/tmp/data.csv",
            "current_csv_path": "/tmp/data.csv",
            "user_query": "Query",
            "step_outputs": {"step_1": df},  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentState(**payload)

        assert "Pandas DataFrame and Series objects cannot cross the Scrygent model boundary." in str(exc_info.value)

    def test_rejects_non_string_user_query(self) -> None:
        """Inject an integer for the `user_query` field.

        The schema must enforce string type for the query to prevent downstream
        string manipulation errors in the Profiler and Planner nodes.
        """
        payload: dict[str, Any] = {
            "original_csv_path": "/tmp/data.csv",
            "current_csv_path": "/tmp/data.csv",
            "user_query": 12345,  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentState(**payload)

        assert "Input should be a valid string" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `session_id` field.

        The `extra="forbid"` rule must apply to prevent schema drift and
        silent acceptance of untracked state variables.
        """
        payload: dict[str, Any] = {
            "original_csv_path": "/tmp/data.csv",
            "current_csv_path": "/tmp/data.csv",
            "user_query": "Query",
            "session_id": "abc-123",
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentState(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

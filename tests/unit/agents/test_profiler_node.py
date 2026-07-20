"""Destructive test suite for the deterministic Profiler Node.

This module aggressively tests the initial profiling entry point. It ensures
that the node correctly orchestrates the pre-flight scrub and profiling
engine, mutates the state to point to the clean CSV, and fails gracefully
with exact error messages if the ingestion pipeline breaks.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scrygent.agents.profiler_node import run_profiler_node
from scrygent.models.state import AgentState


class TestRunProfilerNodeExecution:
    """Tests validating the end-to-end execution and state mutation of the profiler."""

    def test_use_case_profiles_dataset_and_swaps_to_clean_csv_path(self, valid_agent_state: AgentState) -> None:
        """Inject a valid AgentState pointing to a messy CSV on disk.

        Asserts the node successfully orchestrates the pre-flight scrub, profiles
        the data, and updates both `original_csv_path` and `current_csv_path`
        to the newly generated clean temp file path.
        """
        original_path = valid_agent_state.original_csv_path

        result = run_profiler_node(valid_agent_state)

        assert "data_profile" in result
        assert isinstance(result["data_profile"], dict)
        assert "global_schema" in result["data_profile"]
        assert "column_aliases" in result["data_profile"]

        # Verify the paths were swapped to the clean CSV
        assert result["original_csv_path"] != original_path
        assert result["current_csv_path"] == result["original_csv_path"]
        assert Path(result["current_csv_path"]).exists()

    def test_use_case_logs_no_errors_on_success(self, valid_agent_state: AgentState) -> None:
        """Inject a valid AgentState.

        Asserts the node does not append to the `error_log` or set the
        `execution_status` to aborted on a successful run.
        """
        result = run_profiler_node(valid_agent_state)

        assert "error_log" not in result or result["error_log"] == []
        assert "execution_status" not in result


class TestRunProfilerNodeFailures:
    """Tests validating the graceful failure modes and exact error payloads."""

    def test_aborts_with_exact_error_when_file_is_missing(self, valid_agent_state: AgentState) -> None:
        """Inject a state pointing to a non-existent file.

        The node must catch the Pandas FileNotFoundError, append an exact error
        message to `error_log`, and set `execution_status` to `"aborted"`.
        """
        ghost_path = Path("/tmp/scrygent_ghost_file_12345.csv")
        valid_agent_state.original_csv_path = ghost_path

        result = run_profiler_node(valid_agent_state)

        assert result["execution_status"] == "aborted"
        assert len(result["error_log"]) == 1
        # Pandas raises "[Errno 2] No such file or directory: '...'"
        assert result["error_log"][0].startswith(
            f"Profiler initialization failed: [Errno 2] No such file or directory: '{ghost_path}'"
        )

    def test_aborts_with_exact_error_when_preflight_fails(self, valid_agent_state: AgentState) -> None:
        """Mock `preflight_clean_dataset` to raise a simulated `ValueError`.

        The node must catch the internal pipeline failure, format the exact
        error string, and halt the graph gracefully.
        """
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "scrygent.agents.profiler_node.preflight_clean_dataset",
                MagicMock(side_effect=ValueError("Disk corruption detected")),
            )

            result = run_profiler_node(valid_agent_state)

        assert result["execution_status"] == "aborted"
        assert result["error_log"][-1] == "Profiler initialization failed: Disk corruption detected"

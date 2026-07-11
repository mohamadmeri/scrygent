"""Unit tests for planner_node.py"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from scrygent.agents.planner_node import run_planner_node
from scrygent.models.state import AgentState
from scrygent.models.outputs import CSVProfile
from scrygent.models.step_models import Plan, Step
from scrygent.contracts.tool_names import ToolName


@pytest.fixture
def minimal_state():
    """A valid AgentState with a simple data profile."""
    return AgentState(
        original_csv_path=Path("/tmp/test.csv"),
        current_csv_path=Path("/tmp/test.csv"),
        user_query="What is the average revenue?",
        data_profile=CSVProfile(global_schema={"revenue": "float64"}, row_count=100),
    )


@pytest.fixture
def sample_plan():
    """A realistic Plan that the mocked LLM would return."""
    return Plan(steps=[
        Step(
            step_id="1",
            rationale="Fetch detailed stats for revenue",
            tool_name=ToolName.REQUEST_COLUMN_STATS,
            parameters={"columns": ["revenue"]},
        )
    ])


class TestPlannerNode:
    def test_successful_plan_generation(self, mocker, minimal_state, sample_plan):
        """Planner returns plan and updates execution_status to running."""
        mocker.patch(
            "scrygent.agents.planner_node.retrieve_experience",
            return_value="No past experience available.",
        )

        # Create a mock chain whose invoke returns our sample plan
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = sample_plan

        # Mock ChatPromptTemplate.from_messages so its __or__ returns mock_chain
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mocker.patch(
            "scrygent.agents.planner_node.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        )

        # Mock get_structured_llm – it's not used after the prompt mock, but we keep it
        mocker.patch(
            "scrygent.agents.planner_node.get_structured_llm",
            return_value=MagicMock(),
        )

        result = run_planner_node(minimal_state)

        assert "plan" in result
        assert result["plan"] == sample_plan
        assert result["execution_status"] == "running"

    def test_missing_data_profile_aborts(self, mocker):
        """Planner must abort if data_profile is None."""
        state = AgentState(
            original_csv_path=Path("/tmp/test.csv"),
            current_csv_path=Path("/tmp/test.csv"),
            user_query="query",
            data_profile=None,
        )
        result = run_planner_node(state)

        assert result["execution_status"] == "aborted"
        assert len(result["error_log"]) == 1
        assert "data_profile is missing" in result["error_log"][0]

    def test_llm_invocation_failure_aborts(self, mocker, minimal_state):
        """If the LLM chain raises an exception, the node should abort and log the error."""
        mocker.patch(
            "scrygent.agents.planner_node.retrieve_experience",
            return_value="Some experience",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ValueError("LLM timeout")

        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mocker.patch(
            "scrygent.agents.planner_node.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        )
        mocker.patch(
            "scrygent.agents.planner_node.get_structured_llm",
            return_value=MagicMock(),
        )

        result = run_planner_node(minimal_state)

        assert result["execution_status"] == "aborted"
        assert any("Planner failed" in err for err in result["error_log"])

    def test_experience_retrieval_is_used(self, mocker, minimal_state, sample_plan):
        """The node passes experience context into the LLM prompt."""
        mocker.patch(
            "scrygent.agents.planner_node.retrieve_experience",
            return_value="PAST QUERY: ...\nSUCCESSFUL PLAN:\n...",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = sample_plan

        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mocker.patch(
            "scrygent.agents.planner_node.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        )
        mocker.patch(
            "scrygent.agents.planner_node.get_structured_llm",
            return_value=MagicMock(),
        )

        run_planner_node(minimal_state)

        # Ensure the invoke was called with the experience_context
        call_args = mock_chain.invoke.call_args[0][0]
        assert "experience_context" in call_args
        assert call_args["experience_context"] == "PAST QUERY: ...\nSUCCESSFUL PLAN:\n..."

    def test_replan_status_is_preserved(self, mocker, minimal_state, sample_plan):
        """If state has has_replanned=True, the planner still works correctly."""
        minimal_state.has_replanned = True
        mocker.patch(
            "scrygent.agents.planner_node.retrieve_experience",
            return_value="",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = sample_plan

        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        mocker.patch(
            "scrygent.agents.planner_node.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        )
        mocker.patch(
            "scrygent.agents.planner_node.get_structured_llm",
            return_value=MagicMock(),
        )

        result = run_planner_node(minimal_state)
        assert result["execution_status"] == "running"

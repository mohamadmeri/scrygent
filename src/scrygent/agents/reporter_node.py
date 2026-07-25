"""Final synthesis node for the Scrygent compiler.

This node consumes the verified JSON outputs from the deterministic
execution engine and synthesizes the final user-facing report or
benchmark answer. It also commits successful execution plans to
long-term semantic memory.
"""

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..core.config import settings
from ..core.llm_factory import get_structured_llm
from ..core.memory.store import commit_experience
from ..core.resilience import ServiceExhaustedError, resilient_call
from ..models.outputs import AnalysisReport, DirectAnswer
from ..models.state import AgentState
from ..prompts.reporter import EVAL_SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def run_reporter_node(state: AgentState) -> dict[str, Any]:
    """Synthesizes the final response from verified tool outputs.

    Routes to either a full AnalysisReport or a strict DirectAnswer
    based on the AgentState.eval_mode flag. Commits the execution plan
    to semantic memory upon successful completion.

    Args:
        state: The current execution state containing step outputs and the user query.

    Returns:
        A dictionary containing the final_report and the updated execution_status.
    """
    logger.info("--- NODE: REPORTER (Eval Mode: %s) ---", state.eval_mode)

    if not state.step_outputs:
        logger.warning("Reporter invoked but step_outputs is empty. Query might be unanswerable.")

    try:
        # Determine target schema and prompt based on Eval Mode
        target_schema = DirectAnswer if state.eval_mode else AnalysisReport
        system_prompt = EVAL_SYSTEM_PROMPT if state.eval_mode else REPORTER_SYSTEM_PROMPT

        structured_llm = get_structured_llm(
            pydantic_schema=target_schema, model_name=settings.reporter_reasoning_model, method=settings.get_reporter_method
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])

        chain = prompt | structured_llm

        # Serialize step_outputs cleanly to prevent LLM context pollution from Python objects
        outputs_json = json.dumps(state.step_outputs, separators=(",", ":"))
        profile_json = json.dumps(state.data_profile.model_dump(mode="json", exclude_none=True) if state.data_profile else {}, separators=(",", ":"))

        # Safely extract the bidirectional map
        alias_json = json.dumps(state.data_profile.column_aliases if state.data_profile else {}, separators=(",", ":"))

        final_report = resilient_call(
            lambda: chain.invoke({
                "user_query": state.user_query,
                "step_outputs": outputs_json,
                "data_profile": profile_json,
                "column_aliases": alias_json,
            }),
            service="Reporter",
        )

        logger.info("Reporter successfully synthesized the final output.")

        # Commit successful execution plans to long-term semantic memory
        if state.plan:
            commit_experience(state.user_query, state.plan)

        return {"final_report": final_report, "execution_status": "complete"}

    except ServiceExhaustedError as e:
        logger.error("Reporter Node aborted: %s exhausted its retry budget.", e.service)
        return {
            "error_log": state.error_log
            + [f"{e.service} is temporarily unavailable (rate limited after {e.attempts} attempts) while synthesizing the final report."],
            "execution_status": "aborted",
        }
    except Exception as e:
        logger.error("Reporter Node failed: %s", e)
        return {"error_log": state.error_log + [f"Reporter failed: {str(e)}"], "execution_status": "aborted"}

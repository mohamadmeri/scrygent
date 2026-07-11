import logging
import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..models.state import AgentState
from ..models.outputs import AnalysisReport, DirectAnswer
from ..llm_factory import get_structured_llm
from ..resilience import resilient_call, ServiceExhaustedError
from ..prompts.reporter import REPORTER_SYSTEM_PROMPT, EVAL_SYSTEM_PROMPT
from ..memory.store import commit_experience

logger = logging.getLogger(__name__)

def run_reporter_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Node: The Reporter.
    Synthesizes the final response from the verified JSON step_outputs.
    """
    logger.info("--- NODE: REPORTER (Eval Mode: %s) ---", state.eval_mode)

    if not state.step_outputs:
        logger.warning("Reporter invoked but step_outputs is empty. Query might be unanswerable.")

    try:
        # Determine target schema and prompt based on Eval Mode
        target_schema = DirectAnswer if state.eval_mode else AnalysisReport
        system_prompt = EVAL_SYSTEM_PROMPT if state.eval_mode else REPORTER_SYSTEM_PROMPT

        # Initialize OpenRouter LLM bound to the target schema
        structured_llm = get_structured_llm(pydantic_schema=target_schema)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])

        chain = prompt | structured_llm
        
        # Serialize step_outputs cleanly so the LLM doesn't choke on Python objects
        outputs_json = json.dumps(state.step_outputs, indent=2)

        # Invoke the LLM
        final_report = resilient_call(
            lambda: chain.invoke({
                "user_query": state.user_query,
                "step_outputs": outputs_json
            }),
            service="Groq (Reporter)",
        )

        logger.info("Reporter successfully synthesized the final output.")

        if state.plan:
            commit_experience(state.user_query, state.plan)

        # Return the state update
        return {
            "final_report": final_report,
            "execution_status": "complete"
        }

    except ServiceExhaustedError as e:
        logger.error("Reporter Node aborted: %s exhausted its retry budget.", e.service)
        return {
            "error_log": state.error_log + [
                f"{e.service} is temporarily unavailable (rate limited after "
                f"{e.attempts} attempts) while synthesizing the final report."
            ],
            "execution_status": "aborted",
        }
    except Exception as e:
        logger.error("Reporter Node failed: %s", e)
        return {
            "error_log": state.error_log + [f"Reporter failed: {str(e)}"],
            "execution_status": "aborted"
        }

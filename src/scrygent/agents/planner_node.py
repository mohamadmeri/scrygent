import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from ..core.config import settings
from ..core.llm_factory import get_structured_llm
from ..core.memory.store import retrieve_experience
from ..core.resilience import resilient_call
from ..models.abstract_step_models import DraftPlan
from ..models.state import AgentState
from ..models.step_models import Plan
from ..prompts.planner import EMISSION_SYSTEM_PROMPT, EMISSION_USER_TEMPLATE, PARSER_SYSTEM_PROMPT, PARSER_USER_TEMPLATE
from ..prompts.schema_formatter import get_tool_specs

logger = logging.getLogger(__name__)


def run_planner_node(state: AgentState) -> dict[str, Any]:
    """Executes the 2-pass compilation pipeline to generate a strict execution plan.

    Args:
        state: The current execution state containing the data profile and user query.

    Returns:
        A dictionary containing the compiled Plan and the updated execution_status,
        or an abort signal if the compilation pipeline fails.
    """
    logger.info("--- NODE: PLANNER (2-Pass Compiler) ---")

    if state.data_profile is None:
        error_msg = "Planner invoked but data_profile is missing from state."
        logger.error(error_msg)
        return {"error_log": state.error_log + [error_msg], "execution_status": "aborted"}

    try:
        profile_context = state.data_profile.model_dump_json(exclude_none=True)
        experience_context = retrieve_experience(state.user_query)
        tool_specs = get_tool_specs()

        # PASS 1: Reasoning
        draft_llm = get_structured_llm(pydantic_schema=DraftPlan, model_name=settings.reasoning_model)
        # PASS 2: Syntax formatting
        final_llm = get_structured_llm(pydantic_schema=Plan, model_name=settings.formatting_model, method=settings.get_emission_method)

        # PASS 1: PARSER (Generate AST)
        logger.info("Planner Pass 1: Parsing Abstract Intent...")
        parser_prompt = ChatPromptTemplate.from_messages([
            ("system", PARSER_SYSTEM_PROMPT),
            ("user", PARSER_USER_TEMPLATE),
        ])
        draft_plan: DraftPlan = resilient_call(
            lambda: (parser_prompt | draft_llm).invoke({
                "data_profile": profile_context,
                "experience_context": experience_context,
                "query": state.user_query,
            }),
            service="Planner (Parser Pass)",
        )

        # PASS 2: IR EMISSION (Bind to Strict Pydantic Contracts)
        logger.info("Planner Pass 2: Emitting Strict IR JSON...")
        base_emission_prompt = ChatPromptTemplate.from_messages([
            ("system", EMISSION_SYSTEM_PROMPT),
            ("user", EMISSION_USER_TEMPLATE),
        ])

        max_emission_retries = 2
        final_plan = None
        last_error_msg = ""

        draft_json_str = draft_plan.model_dump_json(exclude_none=True)

        for attempt in range(max_emission_retries + 1):
            try:
                if attempt > 0:
                    correction_system = EMISSION_SYSTEM_PROMPT + (
                        f"\n\nCRITICAL CORRECTION REQUIRED:\nYour previous JSON output failed schema validation: {last_error_msg}\n"
                    )
                    current_prompt = ChatPromptTemplate.from_messages([
                        ("system", correction_system),
                        ("user", EMISSION_USER_TEMPLATE),
                    ])
                else:
                    current_prompt = base_emission_prompt

                final_plan = resilient_call(
                    lambda p=current_prompt: (p | final_llm).invoke({  # type: ignore
                        "data_profile": profile_context,
                        "tool_specs": tool_specs,
                        "draft_plan": draft_json_str,
                        "query": state.user_query,
                    }),
                    service="Planner (Emission Pass)",
                )
                break

            except ValidationError as e:
                first_error = e.errors()[0]
                loc = " -> ".join(str(part) for part in first_error["loc"]) if first_error["loc"] else "root"
                msg = first_error.get("msg", "Unknown validation failure")
                last_error_msg = f"Field '{loc}': {msg}"
                logger.warning("Planner Pass 2 IR validation failed: %s", last_error_msg)

                if attempt == max_emission_retries:
                    raise ValueError(f"Planner failed to generate a valid Plan. Last error: {last_error_msg}") from None

        return {"plan": final_plan, "execution_status": "running"}

    except Exception as e:
        logger.error("Planner Node failed during compilation: %s", e, exc_info=True)
        return {"error_log": state.error_log + [f"Compiler Pipeline failed: {str(e)}"], "execution_status": "aborted"}

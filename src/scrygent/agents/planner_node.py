"""Three-pass compiler planner node.

This node implements the core planning logic of Scrygent, translating
natural language queries into strict Intermediate Representation (IR)
plans through a three-phase compilation pipeline:
1. Parser: Extracts logical intent into an Abstract Syntax Tree.
2. Optimizer: Applies execution heuristics to minimize computational cost.
3. Emitter: Binds the optimized plan to strict Pydantic JSON schemas.
"""

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..llm_factory import get_structured_llm
from ..memory.store import retrieve_experience
from ..models.abstract_step_models import DraftPlan
from ..models.state import AgentState
from ..models.step_models import Plan
from ..prompts.planner import EMISSION_SYSTEM_PROMPT, OPTIMIZER_SYSTEM_PROMPT, PARSER_SYSTEM_PROMPT
from ..prompts.schema_formatter import get_tool_specs
from ..resilience import ServiceExhaustedError, resilient_call

logger = logging.getLogger(__name__)


def _dump_plan_debug(draft: DraftPlan, optimized: DraftPlan, final: Plan) -> None:
    """Writes the 3 compiler passes to a local JSON file for auditing."""
    debug_payload = {
        "pass_1_draft": json.loads(draft.model_dump_json()),
        "pass_2_optimized": json.loads(optimized.model_dump_json()),
        "pass_3_final_ir": json.loads(final.model_dump_json()),
    }
    with open("planner_debug.json", "w") as f:
        json.dump(debug_payload, f, indent=2)


def run_planner_node(state: AgentState) -> dict[str, Any]:
    """Executes the 3-pass compilation pipeline to generate a strict execution plan.

    Args:
        state: The current execution state containing the data profile and user query.

    Returns:
        A dictionary containing the compiled Plan and the updated execution_status,
        or an abort signal if the compilation pipeline fails.
    """
    logger.info("--- NODE: PLANNER (3-Pass Compiler) ---")

    if state.data_profile is None:
        error_msg = "Planner invoked but data_profile is missing from state."
        logger.error(error_msg)
        return {"error_log": state.error_log + [error_msg], "execution_status": "aborted"}

    try:
        profile_context = state.data_profile.model_dump_json(indent=2)
        experience_context = retrieve_experience(state.user_query)
        tool_specs = get_tool_specs()

        # Initialize LLMs for Draft and Final passes
        draft_llm = get_structured_llm(pydantic_schema=DraftPlan, provider="groq")
        final_llm = get_structured_llm(pydantic_schema=Plan, provider="groq")

        # PASS 1: PARSER (Generate AST)
        logger.info("Planner Pass 1: Parsing Abstract Intent...")
        parser_prompt = ChatPromptTemplate.from_messages([
            ("system", PARSER_SYSTEM_PROMPT),
            ("user", "USER QUERY: {query}"),
        ])
        draft_plan: DraftPlan = resilient_call(
            lambda: (parser_prompt | draft_llm).invoke({
                "data_profile": profile_context,
                "experience_context": experience_context,
                "query": state.user_query,
            }),
            service="Groq (Parser Pass)",
        )

        # PASS 2: OPTIMIZER (Apply Execution Heuristics)
        logger.info("Planner Pass 2: Optimizing Execution Strategy...")
        optimizer_prompt = ChatPromptTemplate.from_messages([
            ("system", OPTIMIZER_SYSTEM_PROMPT),
            ("user", "Optimize this DraftPlan for the query: {query}"),
        ])
        optimized_plan: DraftPlan = resilient_call(
            lambda: (optimizer_prompt | draft_llm).invoke({
                "data_profile": profile_context,
                "draft_plan": draft_plan.model_dump_json(indent=2),
                "query": state.user_query,
            }),
            service="Groq (Optimizer Pass)",
        )

        # PASS 3: IR EMISSION (Bind to Strict Pydantic Contracts)
        logger.info("Planner Pass 3: Emitting Strict IR JSON...")
        emission_prompt = ChatPromptTemplate.from_messages([
            ("system", EMISSION_SYSTEM_PROMPT),
            ("user", "Translate this Optimized Plan into strict tool payloads for the query: {query}"),
        ])
        final_plan: Plan = resilient_call(
            lambda: (emission_prompt | final_llm).invoke({
                "tool_specs": tool_specs,
                "optimized_plan": optimized_plan.model_dump_json(indent=2),
                "query": state.user_query,
            }),
            service="Groq (Emission Pass)",
        )

        logger.info("3-Pass Compilation Complete. Emitted %d execution steps.", len(final_plan.steps))

        _dump_plan_debug(draft_plan, optimized_plan, final_plan)

        return {"plan": final_plan, "execution_status": "running"}

    except ServiceExhaustedError as e:
        logger.error("Planner Node aborted: %s exhausted its retry budget.", e.service)
        return {
            "error_log": state.error_log
            + [
                f"{e.service} is temporarily unavailable (rate limited after "
                f"{e.attempts} attempts). Please try again shortly."
            ],
            "execution_status": "aborted",
        }
    except Exception as e:
        logger.error("Planner Node failed during compilation: %s", e, exc_info=True)
        return {"error_log": state.error_log + [f"Compiler Pipeline failed: {str(e)}"], "execution_status": "aborted"}

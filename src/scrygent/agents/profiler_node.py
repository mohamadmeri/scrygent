"""Deterministic dataset profiling node.

This node executes the initial data profiling phase before the Planner
is invoked. It extracts the global schema, detailed statistics, and
row samples required for the Planner to generate a mathematically
sound execution plan without guessing data distributions.
"""

import logging
from typing import Any

from ..models.state import AgentState
from ..tools.io import load_csv
from ..tools.profiler import profile_dataframe

logger = logging.getLogger(__name__)


def run_profiler_node(state: AgentState) -> dict[str, Any]:
    """Executes the deterministic profiling pipeline and updates the AgentState.

    Args:
        state: The current execution state containing the CSV path and user query.

    Returns:
        A dictionary containing the updated data_profile or an abort signal
        if the profiling phase fails catastrophically.
    """
    logger.info("--- NODE: PROFILER ---")

    try:
        df = load_csv(state.current_csv_path)
        profile_dict = profile_dataframe(df, state.user_query)
        return {"data_profile": profile_dict}

    except Exception as e:
        logger.error("Profiler Node failed catastrophically: %s", e, exc_info=True)
        return {
            "error_log": state.error_log + [f"Profiler initialization failed: {str(e)}"],
            "execution_status": "aborted",
        }

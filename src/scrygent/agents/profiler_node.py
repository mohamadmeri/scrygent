"""Deterministic dataset profiling node.

This node executes the initial data profiling phase before the Planner
is invoked. It extracts the global schema, detailed statistics, and
row samples required for the Planner to generate a mathematically
sound execution plan without guessing data distributions.
"""

import logging
from typing import Any

from ..core.ingestion import preflight_clean_dataset
from ..models.state import AgentState
from ..tools.io import load_csv
from ..tools.profiler import profile_dataframe

logger = logging.getLogger(__name__)
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
        # 1. Run Pre-Flight Scrub to normalize data and headers
        clean_csv_path, column_aliases = preflight_clean_dataset(state.original_csv_path)

        # 2. Profile the clean dataset
        df = load_csv(clean_csv_path)
        profile_dict = profile_dataframe(df, state.user_query)

        # 3. Attach the bidirectional map
        profile_dict["column_aliases"] = column_aliases

        # 4. Update the state paths so the rest of the graph ONLY uses the clean data
        return {"original_csv_path": clean_csv_path, "current_csv_path": clean_csv_path, "data_profile": profile_dict}

    except Exception as e:
        logger.error("Profiler Node failed catastrophically: %s", e, exc_info=True)
        return {
            "error_log": state.error_log + [f"Profiler initialization failed: {str(e)}"],
            "execution_status": "aborted",
        }

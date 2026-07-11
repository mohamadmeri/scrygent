import logging
from typing import Any

from ..models.state import AgentState
from ..tools.io import load_csv
from ..tools.profiler import profile_dataframe

logger = logging.getLogger(__name__)

def run_profiler_node(state: AgentState) -> dict[str, Any]:
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

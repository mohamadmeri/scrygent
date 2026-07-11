from pydantic import Field

from .filtering import FilterCondition
from ..base_model import ScrygentBaseModel
from ..contracts import NormalizeMethod

class FilterDatasetParams(ScrygentBaseModel):
    filters: list[FilterCondition] = Field(min_length=1)


class NormalizeColumnParams(ScrygentBaseModel):
    column: str
    method: NormalizeMethod


class NoParams(ScrygentBaseModel):
    """
    IR for tools that take no LLM-supplied parameters -- currently only
    reset_dataset, which consumes AgentState.original_csv_path directly.
    The Planner must still emit `"parameters": {}` for these steps; an
    empty dict is the only valid payload, and any stray key here is
    caught the same way a malformed analyze_data payload would be.
    """
    pass


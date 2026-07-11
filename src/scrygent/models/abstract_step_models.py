from pydantic import Field

from ..base_model import ScrygentBaseModel
from ..contracts import ToolName


class AbstractStep(ScrygentBaseModel):
    step_id: str = Field(description="Unique step identifier, e.g., 'step_1'")
    tool_name: ToolName = Field(description="The deterministic tool selected.")
    intent_description: str = Field(description="Plain text description of EXACTLY what this tool must do.")

class DraftPlan(ScrygentBaseModel):
    rationale: str = Field(description="Global analytical strategy based on the data profile.")
    steps: list[AbstractStep] = Field(description="The logical sequence of operations.")

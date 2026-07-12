"""Draft schema for the initial planning phase.

Defines the unstructured, intent-focused payload emitted before the
Optimizer enforces strict IR compliance. Used for early-stage reasoning
where schema validation would prematurely constrain strategic exploration.
"""

from pydantic import Field

from ..base_model import ScrygentBaseModel
from ..contracts import ToolName


class AbstractStep(ScrygentBaseModel):
    """Unstructured step definition capturing intent prior to IR compilation."""

    step_id: str = Field(description="Unique step identifier, e.g., 'step_1'.")
    tool_name: ToolName = Field(description="The deterministic tool selected for this intent.")
    intent_description: str = Field(description="Plain-English description of EXACTLY what this tool must accomplish.")


class DraftPlan(ScrygentBaseModel):
    """Pre-optimization plan capturing global strategy and logical sequence."""

    rationale: str = Field(description="Global analytical strategy based on the data profile.")
    steps: list[AbstractStep] = Field(description="The logical sequence of operations before IR enforcement.")

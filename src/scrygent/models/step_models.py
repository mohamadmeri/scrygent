from typing import Any, Literal

from pydantic import Field, model_validator

from ..base_model import ScrygentBaseModel
from .registry import TOOL_PARAM_MODELS
from ..contracts import ToolName


class Step(ScrygentBaseModel):
    step_id: str = Field(description="Unique identifier for this step.")

    rationale: str = Field(
        description="Describes the purpose of this execution step. Ensures system auditability."
    )

    tool_name: ToolName = Field(
        description="Name of the deterministic tool to execute."
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw tool parameters. Validated and canonicalized against the "
            "registered parameter model for tool_name. After validation, "
            "this dictionary is guaranteed to match the tool's strict IR."
        ),
    )

    required: bool = Field(
        default=True,
        description="If True, failure aborts execution. Otherwise the failure is logged and execution continues.",
    )

    @model_validator(mode="after")
    def _validate_parameters(self) -> "Step":
        # 1. Fetch the strict IR schema from the registry
        try:
            param_model = TOOL_PARAM_MODELS[self.tool_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Internal error: Tool '{self.tool_name}' is not registered "
                "in TOOL_PARAM_MODELS."
            ) from exc

        # 2. Force the raw LLM parameters through the exact IR Pydantic model
        validated = param_model.model_validate(self.parameters)
        
        # 3. Store the clean, canonicalized dictionary back on the Step
        self.parameters = validated.model_dump()

        return self


class StepRecord(ScrygentBaseModel):
    step_id: str = Field(description="Unique step identifier.")
    tool_name: ToolName = Field(description="Tool executed for this step.")
    status: Literal["success", "failed", "skipped"] = Field(
        default="success",
        description="Execution outcome for this step."
    )
    summary: str | None = Field(default=None, description="Short human-readable summary.")
    error: str | None = Field(default=None, description="Error message if the step failed.")
    duration_ms: int | None = Field(default=None, description="Execution time in milliseconds.")


class Plan(ScrygentBaseModel):
    steps: list[Step] = Field(
        description="Ordered list of execution steps."
    )

    @model_validator(mode="after")
    def _lazy_fetch_must_be_sole_step(self) -> "Plan":
        """
        Structural enforcement of PLANNER_SYSTEM_PROMPT directive 1: a Plan
        containing a request_column_stats step must consist of ONLY that
        step.
        """
        has_lazy_fetch = any(s.tool_name == ToolName.REQUEST_COLUMN_STATS for s in self.steps)
        
        if has_lazy_fetch and len(self.steps) > 1:
            raise ValueError(
                "A Plan containing a 'request_column_stats' step must consist "
                "of that single step ONLY. Generate a plan with exactly one "
                "request_column_stats step, or omit it entirely and proceed "
                "using the detailed_stats already available."
            )
        
        return self

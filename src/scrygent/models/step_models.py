"""Execution step and plan schemas for the LangGraph pipeline.

Defines the strict payload structures for plan generation, step execution,
and audit trail recording. Validates LLM-generated parameters against the
IR registry before dispatch.
"""

from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from ..base_model import ScrygentBaseModel
from ..contracts import ToolName
from .registry import TOOL_PARAM_MODELS


class Step(ScrygentBaseModel):
    """Single execution unit within a deterministic plan."""

    step_id: str = Field(description="Unique identifier for this step.")
    rationale: str = Field(description="Purpose of this execution step. Ensures auditability.")
    tool_name: ToolName = Field(description="Deterministic tool to dispatch.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Raw LLM parameters. Canonicalized against the strict IR before dispatch.")
    required: bool = Field(
        default=True,
        description="If True, failure aborts execution. If False, failure is logged and execution continues.",
    )

    @model_validator(mode="after")
    def _validate_parameters(self) -> Step:
        """Validates and canonicalizes raw parameters against the strict IR registry.

        Catches Pydantic validation errors early and re-raises concise,
        actionable messages to prevent verbose tracebacks from leaking into
        state or LLM context.
        """
        try:
            param_model = TOOL_PARAM_MODELS[self.tool_name]
        except KeyError as exc:
            raise RuntimeError(f"Internal registry error: Tool '{self.tool_name}' is not mapped in TOOL_PARAM_MODELS.") from exc

        try:
            validated = param_model.model_validate(self.parameters)
            self.parameters = validated.model_dump()
        except ValidationError as exc:
            # Strip the full Pydantic traceback. Return only the top-level
            # error message and the failing field path.
            first_error = exc.errors()[0]
            loc = " -> ".join(str(part) for part in first_error["loc"]) if first_error["loc"] else "root"
            msg = first_error.get("msg", "Unknown validation failure")
            raise ValueError(f"IR validation failed at '{loc}': {msg}") from None

        return self


class StepRecord(ScrygentBaseModel):
    """Compact audit entry for a completed, failed, or skipped execution step."""

    step_id: str = Field(description="Unique step identifier.")
    tool_name: ToolName = Field(description="Tool dispatched for this step.")
    status: Literal["success", "failed", "skipped"] = Field(default="success", description="Final execution outcome.")
    summary: str | None = Field(default=None, description="Human-readable execution summary.")
    error: str | None = Field(default=None, description="Concise error message if the step failed.")
    duration_ms: int | None = Field(default=None, description="Execution time in milliseconds.")


class Plan(ScrygentBaseModel):
    """Ordered sequence of execution steps emitted by the Planner."""

    steps: list[Step] = Field(description="Strictly typed execution sequence.")

    @model_validator(mode="after")
    def _lazy_fetch_must_be_sole_step(self) -> Plan:
        """Enforces structural constraint: lazy-fetch plans must contain exactly one step.

        Prevents the Planner from attempting complex reasoning alongside
        mid-session profile augmentation, guaranteeing deterministic data routing.
        """
        has_lazy_fetch = any(s.tool_name == ToolName.REQUEST_COLUMN_STATS for s in self.steps)

        if has_lazy_fetch and len(self.steps) > 1:
            raise ValueError(
                "A Plan containing 'request_column_stats' must consist of exactly one step. "
                "Omit all other operations until the statistical profile is resolved."
            )

        return self

from __future__ import annotations
from typing import Any
from pydantic import Field

from ..base_model import ScrygentBaseModel


class CSVProfile(ScrygentBaseModel):
    row_count: int = Field(description="Total number of rows in the dataset. Used to inform query strategies.")
    global_schema: dict[str, str] = Field(description="Column name -> dtype string for EVERY column in the CSV.")
    detailed_stats: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Full statistical metrics for prioritized, high-value columns."
    )
    row_sample: list[dict[str, Any]] = Field(
        default_factory=list,
        description="A 3-row sample with NaN cells replaced by None. For format inference only."
    )
    truncated: bool = Field(
        default=False,
        description="True if detailed_stats covers fewer columns than exist in global_schema. "
        "Signals to the Planner that request_column_stats may be needed."
    )
    missing_detailed_stats: list[str] = Field(
        default_factory=list,
        description=(
            "Columns that exist in global_schema but do NOT have detailed_stats. "
            "Planner MUST resolve this using request_column_stats before relying "
            "on statistical reasoning for these columns."
        )
    )


class PlotMetadata(ScrygentBaseModel):
    file_path: str = Field(description="Disk path to the saved plot image. Never a base64 blob.")
    description: str = Field(description="Short natural-language description of what the plot shows.")


class AnalysisReport(ScrygentBaseModel):
    primary_answer: str = Field(
        description="Direct answer to the user's original query, derived strictly from "
        "verified tool outputs. Must be populated first."
    )
    additional_insights: list[str] | None = Field(
        default=None,
        description="Optional secondary observations surfaced only from tool outputs. "
        "Never sourced from proactive anomaly hallucination."
    )
    plots: list[PlotMetadata] = Field(
        default_factory=list,
        description="File paths and descriptions of visualizations generated during execution."
    )

class DirectAnswer(ScrygentBaseModel):
    answer: str = Field(
        description="The extracted answer value as a scalar, string, boolean, or "
        "comma-separated list, matching benchmark expectations exactly."
    )

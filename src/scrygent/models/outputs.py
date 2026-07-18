"""Output schemas for the deterministic execution engine.

Defines the strict boundaries for profiler results, visualization metadata,
and final reporting payloads.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..base_model import ScrygentBaseModel


class CSVProfile(ScrygentBaseModel):
    """Two-level dataset profile emitted by the Profiler Node."""

    row_count: int = Field(description="Total rows. Used to inform query strategy and sampling limits.")
    global_schema: dict[str, str] = Field(description="Column name to dtype mapping for EVERY column in the CSV.")
    detailed_stats: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Full statistical metrics and structural metadata for prioritized columns."
    )
    row_sample: list[dict[str, Any]] = Field(
        default_factory=list,
        description="A 3-row sample with NaN cells replaced by None. Strictly for format inference.",
    )
    truncated: bool = Field(
        default=False,
        description="True if detailed_stats covers fewer columns than global_schema. "
        "Signals that request_column_stats may be required.",
    )
    missing_detailed_stats: list[str] = Field(
        default_factory=list,
        description="Columns present in global_schema but absent from detailed_stats. "
        "Planner must resolve these before relying on statistical reasoning.",
    )
    query_specific_matches: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Exact string matches extracted from high-cardinality columns based on the user query.",
    )
    regex_skeletons: dict[str, str] = Field(
        default_factory=dict, description="Dominant structural regex patterns for string columns."
    )
    column_aliases: dict[str, str] = Field(
        default_factory=dict, description="Maps clean physical backend columns back to their original UI labels."
    )


class PlotMetadata(ScrygentBaseModel):
    """In-memory visualization reference. Stores Plotly JSON to prevent state memory bloat."""

    plotly_json: str = Field(description="JSON string representation of the interactive Plotly figure.")
    description: str = Field(description="Concise natural-language summary of the visualization.")


class AnalysisReport(ScrygentBaseModel):
    """Final synthesized output for standard query execution."""

    primary_answer: str = Field(
        description="Direct answer to the original query, sourced exclusively from verified tool outputs."
    )
    additional_insights: list[str] | None = Field(
        default=None,
        description="Optional secondary observations surfaced only from tool outputs. "
        "Never sourced from proactive anomaly hallucination.",
    )
    plots: list[PlotMetadata] = Field(
        default_factory=list,
        description="Plotly JSON payloads and descriptions of visualizations generated during execution.",
    )


class DirectAnswer(ScrygentBaseModel):
    """Benchmark-mode output. Contains only the extracted answer value."""

    answer: str = Field(
        description="The exact extracted answer value. Matches benchmark evaluation harness formatting."
    )

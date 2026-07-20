"""Intermediate Representation for data visualization operations.

Defines the strict payload for generating plots, enforcing column-count
arity constraints at the schema boundary before execution.
"""

from pydantic import Field, model_validator

from ..base_model import ScrygentBaseModel
from ..contracts import PlotType

_SINGLE_COLUMN = {PlotType.HISTOGRAM, PlotType.BOX}
_PAIR_COLUMN = {PlotType.BAR, PlotType.LINE, PlotType.SCATTER}


class PlotParams(ScrygentBaseModel):
    """IR for generating data visualizations.

    Enforces column-count arity per plot_type at the schema level.
    Whether the named columns exist in the dataset is validated later
    during tool execution.
    """

    plot_type: PlotType = Field(description="The type of chart to generate.")
    columns: list[str] = Field(min_length=1, description="The columns to plot.")
    title: str | None = Field(default=None, description="Optional title for the plot.")

    @model_validator(mode="after")
    def _arity_matches_plot_type(self) -> PlotParams:
        """Validates that the number of columns matches the plot type requirements."""
        n = len(self.columns)

        if self.plot_type in _SINGLE_COLUMN and n != 1:
            raise ValueError(f"{self.plot_type} requires exactly 1 column, got {n}.")

        if self.plot_type in _PAIR_COLUMN and n != 2:
            raise ValueError(f"{self.plot_type} requires exactly 2 columns, got {n}.")

        if self.plot_type == PlotType.HEATMAP and n < 2:
            raise ValueError(f"heatmap requires at least 2 columns, got {n}.")

        return self

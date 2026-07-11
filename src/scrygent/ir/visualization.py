from pydantic import Field, model_validator

from ..base_model import ScrygentBaseModel
from ..contracts import PlotType

_SINGLE_COLUMN = {PlotType.HISTOGRAM, PlotType.BOX}
_PAIR_COLUMN = {PlotType.BAR, PlotType.LINE, PlotType.SCATTER}
# heatmap: 2+ columns, checked separately (open-ended, not a fixed arity)


class PlotParams(ScrygentBaseModel):
    """
    Column-count arity per plot_type is enforced here, at the IR level --
    it's fully determined by plot_type and len(columns) alone, with no
    dataset access required, so it belongs with the rest of Pydantic's
    structural checks rather than deferred to generate_plot. Whether the
    *named* columns exist in the dataset remains a tool-level (Type B)
    check, since that needs the schema.
    """
    plot_type: PlotType
    columns: list[str] = Field(min_length=1)
    title: str | None = Field(default=None)

    @model_validator(mode="after")
    def _arity_matches_plot_type(self) -> "PlotParams":
        n = len(self.columns)

        if self.plot_type in _SINGLE_COLUMN and n != 1:
            raise ValueError(f"{self.plot_type} requires exactly 1 column, got {n}.")

        if self.plot_type in _PAIR_COLUMN and n != 2:
            raise ValueError(f"{self.plot_type} requires exactly 2 columns, got {n}.")

        if self.plot_type == PlotType.HEATMAP and n < 2:
            raise ValueError(f"heatmap requires at least 2 columns, got {n}.")

        return self

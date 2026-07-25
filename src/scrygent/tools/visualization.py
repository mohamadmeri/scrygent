"""Deterministic visualization engine.

Generates standard analytical plots using Plotly, returning JSON figures
to prevent state memory bloat and enable interactive UI rendering.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..contracts import PlotType
from ..core.config import settings
from .io import load_csv

logger = logging.getLogger(__name__)
MAX_CATEGORIES = 25
MAX_PLOT_POINTS = settings.max_plot_points  # Protects the JSON state boundary


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validates that all specified columns exist in the DataFrame."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}. Available: {list(df.columns)}")


def _plot_bar(df: pd.DataFrame, columns: list[str], title: str | None) -> tuple[go.Figure, str]:
    if len(columns) != 2:
        raise ValueError("bar plot requires exactly 2 columns: [category_column, value_column].")
    cat_col, val_col = columns
    grouped = df.groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(MAX_CATEGORIES).reset_index()
    fig = px.bar(grouped, x=cat_col, y=val_col, title=title)
    suffix = f" (top {MAX_CATEGORIES} categories)" if df[cat_col].nunique() > MAX_CATEGORIES else ""
    return fig, f"Bar chart of mean {val_col} grouped by {cat_col}{suffix}"


def _plot_line(df: pd.DataFrame, columns: list[str], title: str | None) -> tuple[go.Figure, str]:
    if len(columns) != 2:
        raise ValueError("line plot requires exactly 2 columns: [x_column, y_column].")
    x_col, y_col = columns
    ordered = df[[x_col, y_col]].dropna().sort_values(x_col)

    # For lines, we truncate or step-sample to preserve the temporal/ordered trend
    if len(ordered) > MAX_PLOT_POINTS:
        # e.g., take every Nth row to thin the line uniformly
        step = len(ordered) // MAX_PLOT_POINTS
        ordered = ordered.iloc[::step]

    # WEBGL: Force GPU rendering
    fig = px.line(ordered, x=x_col, y=y_col, title=title, render_mode="webgl")
    return fig, f"Line chart of {y_col} over {x_col}"


def _plot_scatter(df: pd.DataFrame, columns: list[str], title: str | None) -> tuple[go.Figure, str]:
    if len(columns) != 2:
        raise ValueError("scatter plot requires exactly 2 columns: [x_column, y_column].")

    x_col, y_col = columns

    # SILVER BULLET: Downsample to protect JSON serialization bloat
    plot_df = df if len(df) <= MAX_PLOT_POINTS else df.sample(n=MAX_PLOT_POINTS, random_state=42)

    # WEBGL: Force GPU rendering
    fig = px.scatter(plot_df, x=x_col, y=y_col, title=title, render_mode="webgl")

    suffix = f" (Sampled {MAX_PLOT_POINTS} points)" if len(df) > MAX_PLOT_POINTS else ""
    return fig, f"Scatter plot of {y_col} vs {x_col}{suffix}"


def _plot_histogram(df: pd.DataFrame, columns: list[str], title: str | None) -> tuple[go.Figure, str]:
    if len(columns) != 1:
        raise ValueError("histogram requires exactly 1 column.")
    col = columns[0]

    # px.histogram passes raw data to JS. Downsample massive datasets.
    plot_df = df if len(df) <= MAX_PLOT_POINTS else df.sample(n=MAX_PLOT_POINTS, random_state=42)

    fig = px.histogram(plot_df, x=col, nbins=30, title=title)
    return fig, f"Histogram of {col}"


def _plot_box(df: pd.DataFrame, columns: list[str], title: str | None) -> tuple[go.Figure, str]:
    if len(columns) != 1:
        raise ValueError("box plot requires exactly 1 column.")
    col = columns[0]

    # Box plots also pass raw data to JS to calculate whiskers. Downsample.
    plot_df = df if len(df) <= MAX_PLOT_POINTS else df.sample(n=MAX_PLOT_POINTS, random_state=42)

    fig = px.box(plot_df, y=col, title=title)
    return fig, f"Box plot of {col}"


def _plot_heatmap(df: pd.DataFrame, columns: list[str], title: str | None) -> tuple[go.Figure, str]:
    if len(columns) < 2:
        raise ValueError("heatmap requires at least 2 columns.")
    corr = df[columns].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title=title or "Correlation Heatmap")
    return fig, f"Correlation heatmap across {len(columns)} columns"


_PLOT_HANDLERS: dict[PlotType, Callable[[pd.DataFrame, list[str], str | None], tuple[go.Figure, str]]] = {
    PlotType.BAR: _plot_bar,
    PlotType.LINE: _plot_line,
    PlotType.SCATTER: _plot_scatter,
    PlotType.HISTOGRAM: _plot_histogram,
    PlotType.BOX: _plot_box,
    PlotType.HEATMAP: _plot_heatmap,
}


def generate_plot(
    current_csv_path: Path,
    plot_type: str,
    columns: list[str],
    title: str | None = None,
) -> dict[str, Any]:
    """Generates a visualization and returns it as a Plotly JSON string."""
    try:
        resolved_type = PlotType(plot_type)
    except ValueError:
        valid = sorted(m.value for m in PlotType)
        raise ValueError(f"Unsupported plot type '{plot_type}'. Choose from: {valid}") from None

    if not columns:
        raise ValueError("generate_plot requires at least 1 column.")

    logger.info("Executing generate_plot | type: %s | columns: %s", resolved_type, columns)
    df = load_csv(current_csv_path)
    _require_columns(df, columns)

    fig, description = _PLOT_HANDLERS[resolved_type](df, columns, title)

    # Apply a clean, professional theme matching Scrygent's dark mode
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )

    return {
        "plotly_json": fig.to_json(),
        "description": description,
    }

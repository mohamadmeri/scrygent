import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ..contracts.visualization import PlotType
from .io import load_csv, write_temp_file

logger = logging.getLogger(__name__)

# Maximum number of categories to display in bar plots
MAX_CATEGORIES = 25


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}. Available: {list(df.columns)}")


def _plot_bar(df: pd.DataFrame, columns: list[str], ax) -> str:
    if len(columns) != 2:
        raise ValueError("bar plot requires exactly 2 columns: [category_column, value_column].")
    cat_col, val_col = columns
    grouped = df.groupby(cat_col)[val_col].mean().sort_values(ascending=False).head(MAX_CATEGORIES)
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(f"mean({val_col})")
    return f"Bar chart of mean {val_col} grouped by {cat_col}" + (
        f" (top {MAX_CATEGORIES} categories)" if df[cat_col].nunique() > MAX_CATEGORIES else ""
    )


def _plot_line(df: pd.DataFrame, columns: list[str], ax) -> str:
    if len(columns) != 2:
        raise ValueError("line plot requires exactly 2 columns: [x_column, y_column].")
    x_col, y_col = columns
    ordered = df[[x_col, y_col]].dropna().sort_values(x_col)
    ax.plot(ordered[x_col], ordered[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return f"Line chart of {y_col} over {x_col}"


def _plot_scatter(df: pd.DataFrame, columns: list[str], ax) -> str:
    if len(columns) != 2:
        raise ValueError("scatter plot requires exactly 2 columns: [x_column, y_column].")
    x_col, y_col = columns
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=15)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return f"Scatter plot of {y_col} vs {x_col}"


def _plot_histogram(df: pd.DataFrame, columns: list[str], ax) -> str:
    if len(columns) != 1:
        raise ValueError("histogram requires exactly 1 column.")
    col = columns[0]
    ax.hist(df[col].dropna(), bins=30)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    return f"Histogram of {col}"


def _plot_box(df: pd.DataFrame, columns: list[str], ax) -> str:
    if len(columns) != 1:
        raise ValueError("box plot requires exactly 1 column.")
    col = columns[0]
    ax.boxplot(df[col].dropna(), orientation="vertical")
    ax.set_ylabel(col)
    ax.set_xticklabels([col])
    return f"Box plot of {col}"


def _plot_heatmap(df: pd.DataFrame, columns: list[str], ax) -> str:
    if len(columns) < 2:
        raise ValueError("heatmap requires at least 2 columns.")
    corr = df[columns].corr(numeric_only=True)
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return f"Correlation heatmap across {len(columns)} columns"


_PLOT_HANDLERS: dict[PlotType, Callable[[pd.DataFrame, list[str], Any], str]] = {
    PlotType.BAR: _plot_bar,
    PlotType.LINE: _plot_line,
    PlotType.SCATTER: _plot_scatter,
    PlotType.HISTOGRAM: _plot_histogram,
    PlotType.BOX: _plot_box,
    PlotType.HEATMAP: _plot_heatmap,
}

SUPPORTED_PLOT_TYPES = set(_PLOT_HANDLERS)


def generate_plot(
    current_csv_path: Path,
    plot_type: str,
    columns: list[str],
    title: str | None = None,
) -> dict[str, Any]:
    raw_plot_type = plot_type
    try:
        plot_type = PlotType(plot_type)
    except ValueError:
        raise ValueError(
            f"Unsupported plot type '{raw_plot_type}'. Choose from: {sorted(SUPPORTED_PLOT_TYPES)}"
        ) from None

    if not columns:
        raise ValueError("generate_plot requires at least 1 column.")

    logger.info("Executing generate_plot | type: %s | columns: %s", plot_type, columns)

    df = load_csv(current_csv_path)
    _require_columns(df, columns)

    numeric_check_cols = columns[1:] if plot_type == PlotType.BAR else columns
    non_numeric = [c for c in numeric_check_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Plot type '{plot_type}' requires numeric column(s); non-numeric: {non_numeric}")

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        description = _PLOT_HANDLERS[plot_type](df, columns, ax)
        if title:
            ax.set_title(title)
            description = f"{title} — {description}"
        fig.tight_layout()

        out_path = write_temp_file(suffix=".png", prefix="scrygent_plot_")
        fig.savefig(out_path, dpi=100)
    finally:
        plt.close(fig)

    return {"file_path": str(out_path), "description": description}

"""Generates sample plots for all supported visualization types.

Saves the resulting PNGs to a local directory for manual visual inspection
of the deterministic rendering engine.
"""

import shutil
from pathlib import Path

import pandas as pd

from scrygent.tools.visualization import generate_plot


def main() -> None:
    """Entry point of the script."""
    output_dir = Path("scripts/plot_outputs")
    output_dir.mkdir(exist_ok=True)

    # Create a numeric-only DataFrame for line/scatter/histogram/box/heatmap
    num_df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2.2, 4.1, 0.9, 5.2, 3.0],
        "z": [5, 4, 3, 2, 1],
        "w": [10, 20, 15, 25, 30],
    })
    num_csv = output_dir / "numeric_data.csv"
    num_df.to_csv(num_csv, index=False)

    # Create a mixed DataFrame for bar (categorical x + numeric y)
    bar_df = pd.DataFrame({
        "category": ["A", "B", "C", "A", "B"],
        "value": [100, 200, 150, 120, 220],
    })
    bar_csv = output_dir / "bar_data.csv"
    bar_df.to_csv(bar_csv, index=False)

    plot_specs = [
        ("bar", ["category", "value"], bar_csv),
        ("line", ["x", "y"], num_csv),
        ("scatter", ["x", "y"], num_csv),
        ("histogram", ["y"], num_csv),
        ("box", ["w"], num_csv),
        ("heatmap", ["x", "y", "z", "w"], num_csv),
    ]

    for ptype, cols, csv_path in plot_specs:
        result = generate_plot(Path(csv_path), plot_type=ptype, columns=cols, title=f"Test {ptype}")
        temp_path = Path(result["file_path"])
        dest_path = output_dir / f"{ptype}.png"
        shutil.copy(temp_path, dest_path)
        print(f"{ptype}: saved to {dest_path}")

    print("\nAll plots generated. Open 'scripts/plot_outputs/' to inspect.")


if __name__ == "__main__":
    main()

"""Generates poisoned variants of a clean dataset for robustness testing.

Creates files with mixed types, alternative delimiters, and non-standard
encodings to ensure the Profiler and Executor fail gracefully with
actionable error messages rather than crashing silently.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def poison_dataset(input_path: Path, out_dir: Path) -> None:
    """Takes a clean dataset and adds mixed types, alternative delimiters, and non-standard encodings.

    Parameters
    ----------
    input_path : Path
    out_dir : Path

    Raises:
    ------
    FileNotFoundError
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_path}")

    df = pd.read_csv(input_path)
    base_name = input_path.stem

    # 1. String artifacts in a numeric column
    df_art = df.copy()
    numeric_cols = df_art.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        target_col = numeric_cols[0]
        df_art[target_col] = df_art[target_col].astype(object)
        # Add currency symbols and N/A strings
        df_art.loc[0:10, target_col] = "$" + df_art[target_col][0:11].astype(str)
        df_art.loc[11:20, target_col] = "N/A"

        out_art = out_dir / f"{base_name}_artifacts.csv"
        df_art.to_csv(out_art, index=False)
        logger.info("Created artifact-poisoned dataset: %s", out_art)

    # 2. Semicolon delimited (despite .csv extension)
    out_semi = out_dir / f"{base_name}_semicolon.csv"
    df.to_csv(out_semi, sep=";", index=False)
    logger.info("Created semicolon-delimited dataset: %s", out_semi)

    # 3. UTF-16 Encoding
    out_utf16 = out_dir / f"{base_name}_utf16.csv"
    df.to_csv(out_utf16, encoding="utf-16", index=False)
    logger.info("Created UTF-16 encoded dataset: %s", out_utf16)


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(description="Generate poisoned datasets for robustness testing.")
    parser.add_argument("input", type=Path, help="Path to the clean input CSV.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("data/poisoned"), help="Directory to save poisoned files."
    )
    args = parser.parse_args()

    poison_dataset(args.input, args.output_dir)


if __name__ == "__main__":
    main()

"""Data ingestion and sanitization gateway.

Handles the transformation of raw, untrusted external data into
strict, normalized formats safe for the Scrygent deterministic engine.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from ..tools.io import write_temp_csv

logger = logging.getLogger(__name__)


def preflight_clean_dataset(input_path: Path) -> tuple[Path, dict[str, str]]:
    """Standard data normalization and cleaning of data and column names.

    Performs Phase 0 data sanitization:
    1. Coerces common string nulls to true NaNs.
    2. Strips leading/trailing whitespace from all string columns.
    3. Normalizes column headers to strict snake_case identifiers.
    4. Resolves duplicate column names.

    Returns:
        Tuple of (Path_to_clean_csv, dictionary mapping physical_name -> original_name)
    """
    logger.info("Executing Pre-Flight Dataset Scrub on %s", input_path)

    # 1. Broad NaN coercion
    missing_values = ["N/A", "n/a", "?", "-", "null", "NULL", ""]
    df = pd.read_csv(input_path, na_values=missing_values, keep_default_na=True)

    # 2. Whitespace stripping for string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. Column Normalization & Collision Resolution
    column_aliases = {}
    new_columns = []
    seen = set()

    for orig_col in df.columns:
        # Lowercase, replace non-alphanumeric with underscores, strip ends
        clean_name = re.sub(r"[^a-z0-9]+", "_", str(orig_col).lower()).strip("_")
        if not clean_name:
            clean_name = "column"

        # Handle duplicates
        final_name = clean_name
        counter = 1
        while final_name in seen:
            final_name = f"{clean_name}_{counter}"
            counter += 1

        seen.add(final_name)
        new_columns.append(final_name)

        # Physical -> Logical Mapping
        column_aliases[final_name] = str(orig_col)

    df.columns = pd.Index(new_columns)

    # 4. Save the pristine dataset to a new temp path
    clean_path = write_temp_csv(df, prefix="scrygent_clean_")

    return clean_path, column_aliases

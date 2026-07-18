"""Core I/O utilities for CSV loading, sampling, and temporary file management.

Provides the foundational disk-boundary functions consumed by the Profiler
and all state-mutating wrangling tools.
"""

import logging
import os
import re
import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd

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


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Loads a CSV file from disk into a Pandas DataFrame.

    Raises:
        FileNotFoundError: If the target path does not exist.
        ValueError: If the file cannot be parsed by the Pandas engine.
    """
    path = Path(file_path)
    logger.info("Attempting to load CSV file from path: %s", path)

    if not path.exists():
        logger.error("File validation failed. Path does not exist: %s", path)
        raise FileNotFoundError(f"Target CSV file does not exist at path: '{path}'")

    try:
        df = pd.read_csv(path)
        logger.info(
            "Successfully loaded CSV. Shape: %s, Memory usage: %d bytes",
            df.shape,
            df.memory_usage().sum(),
        )
        return df
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        logger.error("Pandas parsing engine failed for file: %s. Error: %s", path, e)
        raise ValueError(f"Failed to parse CSV file at '{path}'.") from None


def get_column_sample(df: pd.DataFrame, n: int = 3) -> list[dict[Hashable, Any]]:
    """Extracts a strictly bounded row sample for LLM formatting context.

    Replaces NaN values with None to ensure strict JSON compatibility.
    """
    if df.empty:
        return []

    head = df.head(n)
    safe_df = head.astype(object).where(pd.notna(head), None)
    return safe_df.to_dict(orient="records")


def write_temp_file(suffix: str, prefix: str = "scrygent_") -> Path:
    """Reserves a new temporary file path with the given suffix.

    Returns the Path object without writing content. Callers are
    responsible for writing to the returned path.
    """
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(raw_path)


def write_temp_csv(df: pd.DataFrame, prefix: str = "scrygent_") -> Path:
    """Writes a DataFrame to a new temporary CSV and returns its path.

    Handles empty DataFrames gracefully by writing an empty file.
    """
    path = write_temp_file(suffix=".csv", prefix=prefix)

    if df.empty and len(df.columns) == 0:
        path.write_bytes(b"")
    else:
        df.to_csv(path, index=False)

    logger.info(
        "Wrote transformed dataset to temp path: %s (rows: %d)",
        path,
        len(df),
    )
    return path

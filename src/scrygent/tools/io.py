"""Core I/O utilities for CSV loading, sampling, and temporary file management.

Provides the foundational disk-boundary functions consumed by the Profiler
and all state-mutating wrangling tools.
"""

import logging
import os
import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


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

    Replaces NaN values with None and truncates long strings to prevent
    prompt window bloat from unstructured text columns (e.g., transcripts).
    """
    if df.empty:
        return []

    head = df.head(n)
    safe_df = head.astype(object).where(pd.notna(head), None)

    # Truncate long strings in the sample to prevent token explosion
    max_sample_string_length = 200
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].apply(
            lambda x: f"{str(x)[:max_sample_string_length]}...[truncated]" if isinstance(x, str) and len(x) > max_sample_string_length else x
        )

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

from pathlib import Path
import logging
import pandas as pd
import tempfile
import os

logger = logging.getLogger(__name__)


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Loads a CSV file from disk into a pandas DataFrame."""
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
        raise ValueError(f"Failed to parse CSV file at '{path}'.") from e


def get_column_sample(df: pd.DataFrame, n: int = 3) -> list[dict]:
    """Extracts a strictly bounded row sample for LLM formatting context."""
    if df.empty:
        return []
    
    # Replace NaN with None so it becomes a valid JSON null
    head = df.head(n)
    safe_df = head.astype(object).where(pd.notna(head), None)
    return safe_df.to_dict(orient="records")


def write_temp_file(suffix: str, prefix: str = "scrygent_") -> Path:
    """
    Reserves a new temp file path with the given suffix and returns it,
    without writing content. Callers write to the returned path themselves
    (pandas .to_csv, matplotlib .savefig, etc.) -- this function's only
    job is picking a safe, uniquely-named location, so every tool that
    writes a temp artifact does it the same way.
    """
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(raw_path)


def write_temp_csv(df: pd.DataFrame, prefix: str = "scrygent_") -> Path:
    """Writes a DataFrame to a new temp CSV and returns its path."""
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

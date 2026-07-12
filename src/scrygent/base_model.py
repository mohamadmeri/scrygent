"""Hermetic JSON boundary and base model for the Scrygent compiler.

This module defines the strict serialization boundary between the
deterministic Pandas/NumPy execution engine and the non-deterministic
LLM planning layer. It ensures that no Pandas C-types, NaNs, or
infinite values cross into the JSON state, preventing downstream
API serialization failures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class SanitizationError(ValueError):
    """Raised when data cannot cross the JSON-safe boundary."""


def _sanitize_scalar(value: Any) -> Any:
    """Converts Pandas, NumPy, and standard C-types to native Python primitives.

    This function enforces strict JSON compatibility by mapping non-standard
    types to their closest native equivalents or None.
    """
    if value is pd.NaT:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()

    if isinstance(value, np.str_):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        c = float(value)
        return None if np.isnan(c) or np.isinf(c) else c

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return None if np.isnan(value) or np.isinf(value) else value

    if value is None or isinstance(value, (str, int, bool)):
        return value

    # Fallback for any other object that might be considered NA by Pandas
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    raise SanitizationError(f"Value of type {type(value).__name__!r} ({value!r}) has no sanitization rule.")


def _recursive_sanitize(data: Any) -> Any:
    """Recursively walks collections to sanitize all nested items.

    Prevents DataFrames and Series from crossing the model boundary,
    enforcing the stateless-tool and strict-IR architectural constraints.
    """
    if isinstance(data, BaseModel):
        return data

    if isinstance(data, (pd.DataFrame, pd.Series)):
        raise SanitizationError("Pandas DataFrame and Series objects cannot cross the Scrygent model boundary.")

    if isinstance(data, dict):
        clean: dict[Any, Any] = {}
        for k, v in data.items():
            ck = _sanitize_scalar(k)
            if not isinstance(ck, (str, int, float, bool, type(None))):
                raise SanitizationError(f"Dictionary key {ck!r} is not JSON compatible.")
            clean[ck] = _recursive_sanitize(v)
        return clean

    if isinstance(data, (list, tuple)):
        return [_recursive_sanitize(x) for x in data]

    if isinstance(data, np.ndarray):
        return _recursive_sanitize(data.tolist())

    return _sanitize_scalar(data)


class ScrygentBaseModel(BaseModel):
    """Base model for all Scrygent state and IR schemas.

    Enforces strict JSON compatibility at the instantiation boundary by
    recursively sanitizing all input data before Pydantic field validation.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="wrap")
    @classmethod
    def _sanitize_input(cls, data: Any, handler: Any) -> Any:
        """Sanitizes the entire payload layout before Pydantic processes it."""
        try:
            if isinstance(data, dict):
                clean = _recursive_sanitize(data)
            else:
                clean = data
        except SanitizationError:
            raise
        except Exception as exc:
            raise SanitizationError(f"Unexpected sanitization failure: {exc}") from exc

        return handler(clean)

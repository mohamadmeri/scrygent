from __future__ import annotations
import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator
from typing import Any


def _recursive_sanitize(data: Any) -> Any:
    """Convert numpy/pandas scalars to native Python, NaN/Inf to None."""
    if isinstance(data, pd.Timestamp):
        return data.isoformat()
    if isinstance(data, dict):
        return {k: _recursive_sanitize(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_recursive_sanitize(v) for v in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        f = float(data)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.ndarray):
        return _recursive_sanitize(data.tolist())
    if pd.isna(data):
        return None
    return data


class ScrygentBaseModel(BaseModel):
    """
    Pydantic base class that recursively sanitises all input data.
    Every model in scrygent must inherit from this class to guarantee
    JSON compatibility.
    """

    @model_validator(mode='wrap')
    @classmethod
    def _sanitize_input(cls, data: Any, handler):
        # Apply the recursive sanitizer to the entire input payload.
        clean = _recursive_sanitize(data)
        return handler(clean)

"""Shared fixtures for the deterministic tools test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a DataFrame with mixed types, nulls, and string values for edge cases."""
    return pd.DataFrame({
        "passenger_id": list(range(1, 12)),  # 11 items: 1 to 11
        "age": [22.0, 38.0, np.nan, 35.0, 40.0, 29.0, 50.0, 18.0, 25.0, 33.0, 42.0],
        "gender": ["male"] * 10 + ["female"],  # Highly imbalanced
        "fare": [7.25, 71.28, np.nan, 8.05, 15.0, 12.0, 50.0, 9.0, 10.0, 20.0, 30.0],
        "embarked": ["S", "C", "S", "Q", "S", "C", "Q", "S", "C", "S", "C"],
    })

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from scrygent.tools.profiler import (
    _get_global_schema,
    _extract_query_columns,
    _select_priority_columns,
    _compute_detailed_stats,
    profile_dataframe
)


# --- FIXTURES ---

@pytest.fixture
def sample_df():
    """Provides a dataframe with mixed types and missing values for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "revenue": [100.5, 200.0, np.nan, 400.0, 500.0],  # 1 missing
        "category": ["A", "B", "A", None, None],          # 2 missing
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
    })

# --- TESTS FOR _get_global_schema ---

def test_get_global_schema(sample_df):
    schema = _get_global_schema(sample_df)
    assert isinstance(schema, dict)
    assert schema["id"] in ["int64", "Int64"]
    assert schema["revenue"] in ["float64", "Float64"]
    # Supports both legacy object and modern PyArrow-backed string types
    assert schema["category"] in ["object", "str", "string"]
    assert "datetime64" in schema["date"]

# --- TESTS FOR _extract_query_columns ---

def test_extract_query_columns_prevents_false_positives():
    df_columns = ["id", "a", "on", "rev (usd)"]
    # 'id' is inside 'hidden', 'a' is inside 'data', 'on' is inside 'correlation'
    user_query = "Find the hidden data correlation for rev (usd)"
    
    matched = _extract_query_columns(df_columns, user_query)
    
    # Only "rev (usd)" should match. The regex boundaries must block the false positives.
    assert "rev (usd)" in matched
    assert "id" not in matched
    assert "a" not in matched
    assert "on" not in matched


def test_extract_query_columns_case_insensitivity():
    df_columns = ["ReVenUe", "cost"]
    user_query = "What is the revenue?"
    
    matched = _extract_query_columns(df_columns, user_query)
    assert matched == ["ReVenUe"]


# --- TESTS FOR _select_priority_columns ---

def test_select_priority_columns_under_limit(sample_df):
    query_cols = ["revenue"]
    priority = _select_priority_columns(sample_df, query_cols, max_cols=10)
    
    assert len(priority) == 4
    assert priority[0] == "revenue"  # Query columns should be first
    # The rest should be included


def test_select_priority_columns_over_limit_truncates():
    # Create a dummy df with 5 columns
    df = pd.DataFrame({str(i): [] for i in range(5)})
    query_cols = ["0", "1", "2"]
    
    priority = _select_priority_columns(df, query_cols, max_cols=2)
    
    assert len(priority) == 2
    assert priority == ["0", "1"]  # Truncated query columns


def test_select_priority_columns_fills_by_data_density():
    # id has 0 nulls, revenue has 1 null, category has 2 nulls
    df = pd.DataFrame({
        "category": [None, None, "A"],          # 1 valid
        "revenue": [100.0, np.nan, 200.0],      # 2 valid
        "id": [1, 2, 3]                         # 3 valid
    })
    
    # We want max 2 columns, query provides none.
    # It should pick 'id' and 'revenue' because they have the most non-null data.
    priority = _select_priority_columns(df, query_cols=[], max_cols=2)
    
    assert len(priority) == 2
    assert priority == ["id", "revenue"]


# --- TESTS FOR _compute_detailed_stats ---

def test_compute_detailed_stats_metrics(sample_df):
    stats = _compute_detailed_stats(sample_df, ["revenue", "category"])
    
    # Numeric column checks
    assert "revenue" in stats
    assert stats["revenue"]["null_rate"] == 0.2  # 1 null out of 5 rows
    assert stats["revenue"]["min"] == 100.5
    assert stats["revenue"]["max"] == 500.0
    
    # Categorical column checks
    assert "category" in stats
    assert stats["category"]["null_rate"] == 0.4 # 2 nulls out of 5 rows
    assert "min" not in stats["category"]        # Bounds shouldn't exist for objects
    assert "max" not in stats["category"]


def test_compute_detailed_stats_empty_dataframe_division_by_zero_prevention():
    empty_df = pd.DataFrame({"revenue": pd.Series(dtype='float64')})
    stats = _compute_detailed_stats(empty_df, ["revenue"])
    
    assert stats["revenue"]["null_rate"] == 0.0
    assert stats["revenue"]["min"]
    assert stats["revenue"]["max"]


def test_compute_detailed_stats_all_null_numeric_column():
    df = pd.DataFrame({"revenue": [np.nan, np.nan, np.nan]})
    stats = _compute_detailed_stats(df, ["revenue"])
    
    assert stats["revenue"]["null_rate"] == 1.0
    assert stats["revenue"]["min"]
    assert stats["revenue"]["max"]


# --- TESTS FOR profile_dataframe (Orchestrator) ---

@patch("scrygent.tools.profiler.get_column_sample")
def test_profile_dataframe_standard_execution(mock_get_sample, sample_df):
    mock_get_sample.return_value = [{"id": 1}, {"id": 2}, {"id": 3}]
    user_query = "analyze the revenue"
    
    profile = profile_dataframe(sample_df, user_query)
    
    assert "global_schema" in profile
    assert "detailed_stats" in profile
    assert "truncated" in profile
    assert "row_sample" in profile
    
    assert profile["truncated"] is False
    assert "revenue" in profile["detailed_stats"]
    assert profile["row_sample"] == [{"id": 1}, {"id": 2}, {"id": 3}]
    
    mock_get_sample.assert_called_once()


@patch("scrygent.tools.profiler.get_column_sample")
def test_profile_dataframe_empty(mock_get_sample):
    empty_df = pd.DataFrame()
    profile = profile_dataframe(empty_df, "test")
    
    assert profile["global_schema"] == {}
    assert profile["detailed_stats"] == {}
    assert profile["truncated"] is False
    assert profile["row_sample"] == []
    
    # Should exit early, not calling the sample function
    mock_get_sample.assert_not_called()


@patch("scrygent.tools.profiler.get_column_sample")
def test_profile_dataframe_normalizes_integer_columns(mock_get_sample):
    mock_get_sample.return_value = []
    # Headerless CSVs often result in integer column names
    df = pd.DataFrame({0: [100, 200], 1: ["A", "B"]})
    
    # If the str() cast normalization is missing, this will throw a KeyError
    # when attempting to slice df[other_cols] internally.
    profile = profile_dataframe(df, "test")
    
    assert "0" in profile["global_schema"]
    assert "1" in profile["global_schema"]
    assert "0" in profile["detailed_stats"]

def test_profile_empty_dataframe():
    """Empty dataframe returns blank profile (logs warning)."""
    import pandas as pd
    from scrygent.tools.profiler import profile_dataframe
    
    df_empty = pd.DataFrame()
    result = profile_dataframe(df_empty, "test query")
    
    assert result["global_schema"] == {}
    assert result["detailed_stats"] == {}
    assert result["row_sample"] == []
    assert result["truncated"] is False


def test_profile_query_exceeds_max_columns():
    """Query references more columns than MAX_DETAILED_COLUMNS (logs warning)."""
    import pandas as pd
    from scrygent.tools.profiler import profile_dataframe, MAX_DETAILED_COLUMNS
    
    # Create dataframe with 20 columns, all match query
    df = pd.DataFrame({f"col_{i}": range(10) for i in range(20)})
    
    # Query matches all 20 columns by name
    query = " ".join([f"col_{i}" for i in range(20)])
    
    result = profile_dataframe(df, query)
    
    # detailed_stats should be truncated to MAX_DETAILED_COLUMNS
    assert len(result["detailed_stats"]) == MAX_DETAILED_COLUMNS
    assert result["truncated"] is True


def test_profile_truncation_logging():
    """Profile truncation sets truncated=True and logs info."""
    import pandas as pd
    from scrygent.tools.profiler import profile_dataframe
    
    # Create dataframe with 30 columns
    df = pd.DataFrame({f"col_{i}": range(5) for i in range(30)})
    
    # Query matches only one column
    result = profile_dataframe(df, "col_0")
    
    assert result["truncated"] is True
    assert len(result["detailed_stats"]) < len(result["global_schema"])
    assert len(result["global_schema"]) == 30

def test_select_priority_columns_query_cols_fill_limit():
    """When query columns fill max_cols, other_cols is empty (line 87 edge case)."""
    import pandas as pd
    from scrygent.tools.profiler import _select_priority_columns
    
    df = pd.DataFrame({f"col_{i}": range(5) for i in range(10)})
    query_cols = ["col_0", "col_1", "col_2"]
    
    # max_cols=3 means query_cols fills exactly the limit
    priority = _select_priority_columns(df, query_cols, max_cols=3)
    
    assert len(priority) == 3
    assert priority == ["col_0", "col_1", "col_2"]

def test_select_priority_columns_all_columns_are_query_columns():
    """When all dataframe columns are query columns, other_cols is empty (line 87)."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    query_cols = ["col_a", "col_b"]  # Query includes ALL columns in dataframe
    
    # max_cols is larger than the number of columns
    priority = _select_priority_columns(df, query_cols, max_cols=10)
    
    assert len(priority) == 2
    assert priority == ["col_a", "col_b"]

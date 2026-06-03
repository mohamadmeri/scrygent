import pytest
import pandas as pd

from scrygent.tools.io import load_csv, get_column_sample

@pytest.fixture
def sample_csv_file(tmp_path):
    """Fixture to generate a temporary CSV file for isolated testing."""
    data = {
        "A": [1, 2, 3, 4],
        "B": ["apple", "banana", "cherry", "date"],
        "C": [1.1, 2.2, 3.3, 4.4]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_load_csv_success(sample_csv_file):
    """Verifies that load_csv correctly reads a file into a DataFrame."""
    df = load_csv(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 3)
    assert list(df.columns) == ["A", "B", "C"]

def test_load_csv_file_not_found():
    """Verifies that a explicit FileNotFoundError is thrown for invalid paths."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")

def test_get_column_sample_bounds():
    """Verifies get_column_sample handles slicing and returns structured dicts."""
    data = {"col1": [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)
    
    # Test default strict 3-row payload rule
    sample = get_column_sample(df, n=3)
    assert isinstance(sample, list)
    assert len(sample) == 3
    assert sample[0] == {"col1": 10}
    assert sample[2] == {"col1": 30}

def test_get_column_sample_short_dataframe():
    """Verifies get_column_sample handles dataframes with fewer rows than requested."""
    data = {"col1": [10]}
    df = pd.DataFrame(data)
    
    sample = get_column_sample(df, n=3)
    assert len(sample) == 1
    assert sample[0] == {"col1": 10}

def test_load_csv_parsing_error(tmp_path):
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("col1,col2\n1,2\n3,four,5")
    with pytest.raises(ValueError, match="Failed to parse CSV file"):
        load_csv(bad_file)

def test_get_column_sample_empty_dataframe():
    """Verifies that an empty DataFrame yields an empty sample list."""
    df = pd.DataFrame()
    sample = get_column_sample(df, n=3)
    assert isinstance(sample, list)
    assert len(sample) == 0
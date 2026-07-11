"""Tests for io.py: CSV loading, column sampling, and temporary file writing."""
import pandas as pd
import pytest

from pathlib import Path
from scrygent.tools.io import (
    load_csv,
    get_column_sample,
    write_temp_file,
    write_temp_csv,
)


# ── Fixtures ──
@pytest.fixture
def sample_csv(tmp_path):
    """A simple, well-formed CSV file."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def nan_csv(tmp_path):
    """CSV with NaN values for column sample tests."""
    df = pd.DataFrame({"col": [1.0, None, 3.0]})
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── load_csv ──
class TestLoadCSV:
    def test_read_valid_file(self, sample_csv):
        df = load_csv(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["A", "B"]

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/file.csv")

    def test_parsing_error_raises(self, tmp_path):
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("col1,col2\n1,2\n3,four,5")  # uneven columns
        with pytest.raises(ValueError, match="Failed to parse CSV file"):
            load_csv(bad_file)

    def test_empty_csv(self, tmp_path):
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("A,B\n")  # header only
        df = load_csv(empty_file)
        assert df.empty
        assert list(df.columns) == ["A", "B"]

    def test_accepts_pathlib_path(self, tmp_path):
        p = tmp_path / "p.csv"
        pd.DataFrame({"x": [1]}).to_csv(p, index=False)
        df = load_csv(p)
        assert df.shape == (1, 1)


# ── get_column_sample ──
class TestGetColumnSample:
    def test_normal_case(self, nan_csv):
        df = load_csv(nan_csv)
        sample = get_column_sample(df, n=3)
        assert len(sample) == 3
        # second row is None (NaN)
        assert sample[0] == {"col": 1.0}
        assert sample[1] == {"col": None}
        assert sample[2] == {"col": 3.0}

    def test_fewer_rows_than_n(self, nan_csv):
        df = load_csv(nan_csv)
        sample = get_column_sample(df, n=10)
        assert len(sample) == 3

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        out_path = write_temp_csv(df)
        assert out_path.exists()       # a file was created
        out_path.unlink()

    def test_default_n_equals_three(self, nan_csv):
        df = load_csv(nan_csv)
        sample = get_column_sample(df)
        assert len(sample) == 3

    def test_no_nan_returns_original_values(self, sample_csv):
        df = load_csv(sample_csv)
        sample = get_column_sample(df, n=2)
        assert sample[0] == {"A": 1, "B": "x"}


# ── write_temp_file ──
class TestWriteTempFile:
    def test_returns_pathlib_path(self):
        p = write_temp_file(suffix=".csv")
        assert isinstance(p, Path)
        assert p.name.startswith("scrygent_")
        assert p.suffix == ".csv"
        # clean up
        p.unlink()

    def test_creates_unique_files(self):
        p1 = write_temp_file(suffix=".png")
        p2 = write_temp_file(suffix=".png")
        assert p1 != p2
        p1.unlink()
        p2.unlink()

    def test_custom_prefix(self):
        p = write_temp_file(suffix=".txt", prefix="custom_")
        assert p.name.startswith("custom_")
        p.unlink()

    def test_return_is_path_object(self):
        p = write_temp_file(suffix=".png")
        assert isinstance(p, Path)
        p.unlink()

# ── write_temp_csv ──
class TestWriteTempCSV:
    def test_writes_and_reads_back(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        out_path = write_temp_csv(df, prefix="test_")
        assert out_path.exists()
        assert out_path.suffix == ".csv"
        # Read back and verify
        df_read = pd.read_csv(out_path)
        pd.testing.assert_frame_equal(df, df_read)
        out_path.unlink()

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        out_path = write_temp_csv(df)
        assert out_path.exists()
        assert out_path.stat().st_size == 0      # empty file
        out_path.unlink()

    def test_large_dataframe_no_error(self):
        """Ensure no crash for a moderately large DataFrame."""
        df = pd.DataFrame({"col": range(10000)})
        out_path = write_temp_csv(df)
        assert out_path.stat().st_size > 0
        out_path.unlink()

# ── Integration of io functions ──
class TestIOIntegration:
    def test_write_and_load_roundtrip(self):
        original = pd.DataFrame({"num": [1, 2], "txt": ["hello", "world"]})
        p = write_temp_csv(original, prefix="roundtrip_")
        loaded = load_csv(p)
        pd.testing.assert_frame_equal(original, loaded)
        p.unlink()

    def test_sample_after_load(self, sample_csv):
        df = load_csv(sample_csv)
        sample = get_column_sample(df)
        assert len(sample) == 3
        assert "A" in sample[0]

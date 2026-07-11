"""Tests for visualization.generate_plot – all plot types, errors, and edge cases."""
import pandas as pd
import pytest

from pathlib import Path
from scrygent.tools.visualization import generate_plot


# ── Helpers ──
def _make_csv(tmp_path, data: dict, name: str = "data.csv") -> str:
    df = pd.DataFrame(data)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def bar_csv(tmp_path):
    return _make_csv(tmp_path, {
        "category": ["A", "B", "C", "A", "B"],
        "value": [10, 20, 15, 12, 22],
    })

@pytest.fixture
def line_csv(tmp_path):
    return _make_csv(tmp_path, {
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 1, 5, 3],
    })

@pytest.fixture
def scatter_csv(tmp_path):
    return _make_csv(tmp_path, {
        "x": [1.1, 2.5, 3.2, 4.9, 5.0],
        "y": [2.2, 4.1, 0.9, 5.2, 3.0],
    })

@pytest.fixture
def hist_csv(tmp_path):
    return _make_csv(tmp_path, {
        "val": [1, 2, 2, 3, 3, 3, 4, 4, 5],
    })

@pytest.fixture
def box_csv(tmp_path):
    return _make_csv(tmp_path, {
        "data": [10, 20, 30, 40, 50, 60, 100],
    })

@pytest.fixture
def heatmap_csv(tmp_path):
    return _make_csv(tmp_path, {
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
        "c": [2, 3, 4, 5, 6],
    })

# ── Success tests per plot type ──
class TestGeneratePlotSuccess:
    def test_bar_plot(self, bar_csv):
        res = generate_plot(bar_csv, plot_type="bar", columns=["category", "value"], title="Test Bar")
        assert Path(res["file_path"]).exists()
        assert "Bar chart" in res["description"]

    def test_line_plot(self, line_csv):
        res = generate_plot(line_csv, plot_type="line", columns=["x", "y"])
        assert Path(res["file_path"]).exists()
        assert "Line chart" in res["description"]

    def test_scatter_plot(self, scatter_csv):
        res = generate_plot(scatter_csv, plot_type="scatter", columns=["x", "y"])
        assert Path(res["file_path"]).exists()
        assert "Scatter plot" in res["description"]

    def test_histogram(self, hist_csv):
        res = generate_plot(hist_csv, plot_type="histogram", columns=["val"])
        assert Path(res["file_path"]).exists()
        assert "Histogram" in res["description"]

    def test_box_plot(self, box_csv):
        res = generate_plot(box_csv, plot_type="box", columns=["data"])
        assert Path(res["file_path"]).exists()
        assert "Box plot" in res["description"]

    def test_heatmap(self, heatmap_csv):
        res = generate_plot(heatmap_csv, plot_type="heatmap", columns=["a", "b", "c"])
        assert Path(res["file_path"]).exists()
        assert "Correlation heatmap" in res["description"]

    def test_title_in_description(self, bar_csv):
        res = generate_plot(bar_csv, plot_type="bar", columns=["category", "value"], title="Sales")
        assert "Sales —" in res["description"]

    def test_file_returns_png(self, hist_csv):
        res = generate_plot(hist_csv, plot_type="histogram", columns=["val"])
        assert res["file_path"].endswith(".png")

    def test_plot_with_many_categories_trims_bar(self, tmp_path):
        # More than MAX_CATEGORIES (25)
        data = {"cat": [f"cat_{i}" for i in range(30)], "val": range(30)}
        csv_path = _make_csv(tmp_path, data, name="many_cats.csv")
        res = generate_plot(csv_path, plot_type="bar", columns=["cat", "val"])
        assert "top 25" in res["description"]

    def test_bar_allows_non_numeric_category_column(self, tmp_path):
        # First column can be non-numeric
        csv_path = _make_csv(tmp_path, {"name": ["x","y","z"], "score": [1,2,3]})
        res = generate_plot(csv_path, plot_type="bar", columns=["name", "score"])
        assert Path(res["file_path"]).exists()


# ── Error cases ──
class TestGeneratePlotErrors:
    def test_invalid_plot_type(self, bar_csv):
        with pytest.raises(ValueError):
            generate_plot(bar_csv, plot_type="candlestick", columns=["category", "value"])

    def test_empty_columns_list(self, bar_csv):
        with pytest.raises(ValueError, match="at least 1 column"):
            generate_plot(bar_csv, plot_type="bar", columns=[])

    def test_missing_columns(self, bar_csv):
        with pytest.raises(ValueError, match="Column.* not found"):
            generate_plot(bar_csv, plot_type="bar", columns=["missing", "value"])

    def test_wrong_column_count_for_bar(self, bar_csv):
        with pytest.raises(ValueError, match="bar plot requires exactly 2 columns"):
            generate_plot(bar_csv, plot_type="bar", columns=["category"])

    def test_wrong_column_count_for_line(self, line_csv):
        with pytest.raises(ValueError, match="line plot requires exactly 2 columns"):
            generate_plot(line_csv, plot_type="line", columns=["x"])

    def test_wrong_column_count_for_scatter(self, tmp_path):
        # Provide exactly 3 columns as a dict
        csv_path = _make_csv(tmp_path, {"x": [1,2], "y": [3,4], "z": [5,6]}, name="scatter3.csv")
        with pytest.raises(ValueError, match="scatter plot requires exactly 2 columns"):
            generate_plot(csv_path, plot_type="scatter", columns=["x", "y", "z"])

    def test_histogram_requires_1_column(self, tmp_path):
        csv_path = _make_csv(tmp_path, {"a": [1,2], "b": [3,4]}, name="hist2.csv")
        with pytest.raises(ValueError, match="histogram requires exactly 1 column"):
            generate_plot(csv_path, plot_type="histogram", columns=["a", "b"])

    def test_box_requires_1_column(self, tmp_path):
        csv_path = _make_csv(tmp_path, {"a": [1,2], "b": [3,4]}, name="box2.csv")
        with pytest.raises(ValueError, match="box plot requires exactly 1 column"):
            generate_plot(csv_path, plot_type="box", columns=["a", "b"])
        
    def test_heatmap_requires_at_least_2_columns(self, heatmap_csv):
        with pytest.raises(ValueError, match="heatmap requires at least 2 columns"):
            generate_plot(heatmap_csv, plot_type="heatmap", columns=["a"])

    def test_non_numeric_y_column_in_bar(self, tmp_path):
        csv_path = _make_csv(tmp_path, {"cat": ["a","b"], "val": ["x","y"]})
        with pytest.raises(ValueError, match="requires numeric column"):
            generate_plot(csv_path, plot_type="bar", columns=["cat", "val"])

    def test_non_numeric_in_line(self, tmp_path):
        csv_path = _make_csv(tmp_path, {"x": [1,2], "y": ["a","b"]})
        with pytest.raises(ValueError, match="requires numeric column"):
            generate_plot(csv_path, plot_type="line", columns=["x", "y"])

    def test_non_numeric_in_histogram(self, tmp_path):
        csv_path = _make_csv(tmp_path, {"val": ["a","b","c"]})
        with pytest.raises(ValueError, match="requires numeric column"):
            generate_plot(csv_path, plot_type="histogram", columns=["val"])

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            generate_plot("/fake/path.csv", plot_type="histogram", columns=["val"])

"""Tests for statistics tools: correlation, regression, detect_outliers, request_column_stats."""
import pandas as pd
import pytest

from scrygent.tools.statistics import (
    correlation,
    regression,
    detect_outliers,
    request_column_stats,
)


# ── Fixtures ──
@pytest.fixture
def numeric_csv(tmp_path):
    """CSV with purely numeric columns for statistical tests."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
        "c": [2, 3, 4, 5, 6],
        "d": [1, 1, 1, 1, 1],  # constant column for edge cases
        "e": [1, 2, 3, 4, None],  # column with null
    })
    csv_path = tmp_path / "numeric.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mixed_csv(tmp_path):
    """CSV with mixed types (numeric + string)."""
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
        "name": ["a", "b", "c"],
    })
    csv_path = tmp_path / "mixed.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ── Helper to create temp CSV from DataFrame ──
def _make_csv(tmp_path, df, name="temp.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


# ── correlation ──
class TestCorrelation:
    def test_pearson_two_columns(self, numeric_csv):
        result = correlation(numeric_csv, columns=["a", "b"], method="pearson")
        assert result["method"] == "pearson"
        assert result["column_a"] == "a"
        assert result["column_b"] == "b"
        # Perfect negative correlation
        assert result["correlation"] == pytest.approx(-1.0)

    def test_spearman_two_columns(self, numeric_csv):
        result = correlation(numeric_csv, columns=["a", "b"], method="spearman")
        assert result["method"] == "spearman"
        assert result["correlation"] == pytest.approx(-1.0)

    def test_kendall_two_columns(self, numeric_csv):
        result = correlation(numeric_csv, columns=["a", "b"], method="kendall")
        assert result["method"] == "kendall"
        # Kendall tau for reversed ranks should be -1
        assert result["correlation"] == pytest.approx(-1.0)

    def test_matrix_multiple_columns(self, numeric_csv):
        result = correlation(numeric_csv, columns=["a", "b", "c"])
        assert result["method"] == "pearson"  # default
        pairs = result["pairs"]
        assert len(pairs) == 3  # combinations of 3: (a,b), (a,c), (b,c)
        # Check that each pair is present
        cols_in_pairs = {(p["column_a"], p["column_b"]) for p in pairs}
        assert ("a", "b") in cols_in_pairs

    def test_default_method_is_pearson(self, numeric_csv):
        result = correlation(numeric_csv, columns=["a", "b"])
        assert result["method"] == "pearson"

    def test_insufficient_columns_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="at least 2 columns"):
            correlation(numeric_csv, columns=["a"])

    def test_non_numeric_column_raises(self, mixed_csv):
        with pytest.raises(ValueError, match="not numeric"):
            correlation(mixed_csv, columns=["x", "name"])

    def test_column_not_found_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="Column 'z' not found"):
            correlation(numeric_csv, columns=["a", "z"])

    def test_invalid_method_raises(self, numeric_csv):
        with pytest.raises(ValueError):
            correlation(numeric_csv, columns=["a", "b"], method="invalid")


# ── regression ──
class TestRegression:
    def test_simple_linear(self, numeric_csv):
        # c = 2 + 0.5*a? Actually a=[1,2,3,4,5], c=[2,3,4,5,6] => c = a + 1
        result = regression(numeric_csv, target="c", features=["a"])
        assert result["method"] == "linear"
        assert result["target"] == "c"
        coeffs = result["coefficients"]
        assert "a" in coeffs
        assert coeffs["a"] == pytest.approx(1.0, abs=1e-6)
        assert result["intercept"] == pytest.approx(1.0, abs=1e-6)
        assert result["r_squared"] == pytest.approx(1.0)

    def test_multiple_features(self, numeric_csv):
        # Use a and b to predict c; c = a + 0*b + 1? Actually a+b = 6 for all rows except? a+b always 6, c=a+1, so not exactly collinear. We'll just check that output has two coefficients.
        result = regression(numeric_csv, target="c", features=["a", "b"])
        assert "a" in result["coefficients"]
        assert "b" in result["coefficients"]

    def test_not_enough_rows_raises(self, tmp_path):
        df = pd.DataFrame({"target": [1, 2], "x": [3, 4]})
        csv_path = _make_csv(tmp_path, df, "small.csv")
        with pytest.raises(ValueError, match="Insufficient complete rows"):
            regression(csv_path, target="target", features=["x"])

    def test_target_in_features_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="cannot also appear in features"):
            regression(numeric_csv, target="a", features=["a", "b"])

    def test_empty_features_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="at least 1 feature"):
            regression(numeric_csv, target="a", features=[])

    def test_non_numeric_target_raises(self, mixed_csv):
        with pytest.raises(ValueError, match="not numeric"):
            regression(mixed_csv, target="name", features=["x"])

    def test_non_numeric_feature_raises(self, mixed_csv):
        with pytest.raises(ValueError, match="not numeric"):
            regression(mixed_csv, target="x", features=["name"])

    def test_r_squared_with_zero_variance(self, tmp_path):
        # constant target => r_squared = None
        df = pd.DataFrame({"y": [5, 5, 5], "x": [1, 2, 3]})
        csv_path = _make_csv(tmp_path, df)
        result = regression(csv_path, target="y", features=["x"])
        assert result["r_squared"] is None

    def test_output_keys(self, numeric_csv):
        result = regression(numeric_csv, target="a", features=["b", "c"])
        expected_keys = {"method", "target", "features", "row_count", "intercept", "coefficients", "r_squared"}
        assert expected_keys.issubset(result.keys())

    def test_invalid_method_raises(self, numeric_csv):
        with pytest.raises(ValueError):
            regression(numeric_csv, target="a", features=["b"], method="ridge")


# ── outlier detection ──
class TestDetectOutliers:
    def test_iqr_no_outliers(self, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        csv_path = _make_csv(tmp_path, df)
        result = detect_outliers(csv_path, column="val", method="iqr")
        assert result["outlier_count"] == 0
        assert result["method"] == "iqr"

    def test_iqr_with_outlier(self, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        csv_path = _make_csv(tmp_path, df)
        result = detect_outliers(csv_path, column="val", method="iqr")
        assert result["outlier_count"] == 1
        assert "lower_bound" in result["params"]
        assert "upper_bound" in result["params"]

    def test_zscore_no_outliers(self, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        csv_path = _make_csv(tmp_path, df)
        result = detect_outliers(csv_path, column="val", method="z_score")
        assert result["outlier_count"] == 0

    def test_zscore_with_outlier(self, tmp_path):
        df = pd.DataFrame({"val": [0] * 100 + [1000]})
        csv_path = _make_csv(tmp_path, df)
        result = detect_outliers(csv_path, column="val", method="z_score")
        assert result["outlier_count"] == 1

    def test_zscore_constant_column_raises(self, numeric_csv):
        # column 'd' is constant (all 1)
        with pytest.raises(ValueError, match="zero or undefined variance"):
            detect_outliers(numeric_csv, column="d", method="z_score")

    def test_non_numeric_column_raises(self, mixed_csv):
        with pytest.raises(ValueError, match="not numeric"):
            detect_outliers(mixed_csv, column="name")

    def test_column_not_found_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="Column 'z' not found"):
            detect_outliers(numeric_csv, column="z")

    def test_invalid_method_raises(self, numeric_csv):
        with pytest.raises(ValueError):
            detect_outliers(numeric_csv, column="a", method="grubbs")

    def test_max_outlier_examples_respected(self, tmp_path):
        # Create many outliers
        df = pd.DataFrame({"val": list(range(100)) + [1000]*30})
        csv_path = _make_csv(tmp_path, df)
        result = detect_outliers(csv_path, column="val", method="iqr")
        # outlier_examples should be limited to MAX_OUTLIER_EXAMPLES (20)
        assert len(result["outlier_examples"]) <= 20

    def test_nulls_ignored(self, tmp_path):
        df = pd.DataFrame({"val": [1, 2, None, 4, 100]})
        csv_path = _make_csv(tmp_path, df)
        result = detect_outliers(csv_path, column="val", method="iqr")
        # Should not count null as outlier
        assert result["outlier_count"] == 1  # only 100


# ── request_column_stats ──
class TestRequestColumnStats:
    def test_single_column(self, numeric_csv):
        result = request_column_stats(numeric_csv, columns=["a"])
        assert "detailed_stats" in result
        stats = result["detailed_stats"]
        assert "a" in stats
        # Should contain typical metrics (dtype, null_rate, etc.)
        assert "dtype" in stats["a"]

    def test_multiple_columns(self, numeric_csv):
        result = request_column_stats(numeric_csv, columns=["a", "b"])
        assert set(result["detailed_stats"].keys()) == {"a", "b"}

    def test_empty_columns_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="at least 1 column"):
            request_column_stats(numeric_csv, columns=[])

    def test_missing_column_raises(self, numeric_csv):
        with pytest.raises(ValueError, match="Column.* not found"):
            request_column_stats(numeric_csv, columns=["a", "unknown"])

    def test_non_existent_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            request_column_stats("/nonexistent.csv", columns=["a"])

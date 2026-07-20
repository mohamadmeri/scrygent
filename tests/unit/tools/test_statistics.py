"""Destructive and functional test suite for the statistical analysis engine.

This module aggressively tests the deterministic implementations for
correlation, regression, outlier detection, and lazy column profiling.
It ensures that hallucinated columns, non-numeric data, and mathematical
edge cases (like zero variance) are strictly rejected with actionable errors.
"""

from pathlib import Path

import pandas as pd
import pytest

from scrygent.tools.statistics import correlation, detect_outliers, regression, request_column_stats


@pytest.fixture
def constant_column_csv(tmp_path: Path) -> Path:
    """Provide a CSV with a numeric column with zero variance."""
    df = pd.DataFrame({"constant_val": [5, 5, 5, 5], "id": [1, 2, 3, 4]})
    path = tmp_path / "constant.csv"
    df.to_csv(path, index=False)
    return path


class TestCorrelation:
    """Tests validating the deterministic correlation engine."""

    def test_use_case_executes_pairwise_correlation(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for pairwise correlation between `age` and `fare`.

        Asserts the tool computes the Pearson coefficient and returns the exact
        expected schema for a 2-column request.
        """
        result = correlation(dummy_csv_path, columns=["age", "fare"], method="pearson")

        assert result["method"] == "pearson"
        assert result["column_a"] == "age"
        assert result["column_b"] == "fare"
        assert isinstance(result["correlation"], float)

    def test_use_case_executes_matrix_correlation(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for a correlation matrix across 3 columns.

        Asserts the tool returns a list of pairwise dictionaries instead of a
        single coefficient.
        """
        result = correlation(dummy_csv_path, columns=["age", "fare", "passenger_id"], method="pearson")

        assert "pairs" in result
        assert len(result["pairs"]) == 3  # 3 choose 2 pairs

    def test_rejects_less_than_two_columns(self, dummy_csv_path: Path) -> None:
        """Inject a request with only one column.

        The tool must reject this immediately as correlation requires >= 2 columns.
        """
        with pytest.raises(ValueError, match="correlation requires at least 2 columns."):
            correlation(dummy_csv_path, columns=["age"])

    def test_rejects_hallucinated_column_and_provides_available_list(self, dummy_csv_path: Path) -> None:
        """Inject a non-existent column 'fake_col'.

        The tool must raise a ValueError listing the exact available columns
        to fuel the self-healing correction loop.
        """
        with pytest.raises(ValueError, match="Column 'fake_col' not found.") as exc_info:
            correlation(dummy_csv_path, columns=["age", "fake_col"])

        assert "Available: ['passenger_id', 'survived', 'age', 'fare', 'embarked']" in str(exc_info.value)

    def test_rejects_non_numeric_column(self, dummy_csv_path: Path) -> None:
        """Inject a string column ('embarked') into the correlation request.

        The tool must enforce numeric dtypes to prevent Pandas TypeErrors.
        """
        with pytest.raises(ValueError, match="Column 'embarked' is not numeric"):
            correlation(dummy_csv_path, columns=["age", "embarked"])

    def test_rejects_hallucinated_method(self, dummy_csv_path: Path) -> None:
        """Inject an unsupported method like 'covariance'.

        The tool must reject the hallucinated method and list valid options.
        """
        with pytest.raises(ValueError, match="Unsupported correlation method 'covariance'. Choose from:") as exc_info:
            correlation(dummy_csv_path, columns=["age", "fare"], method="covariance")

        assert "'pearson'" in str(exc_info.value)


class TestRegression:
    """Tests validating the deterministic linear regression engine."""

    def test_use_case_executes_linear_regression(self, dummy_csv_path: Path) -> None:
        """Inject a valid request to predict `fare` from `age`.

        Asserts the tool computes the OLS model and returns the exact schema
        containing intercept, coefficients, and r_squared.
        """
        result = regression(
            dummy_csv_path,
            target="fare",
            features=["age"],
            method="linear",
        )

        assert result["method"] == "linear"
        assert result["target"] == "fare"
        assert result["features"] == ["age"]
        assert "intercept" in result
        assert "coefficients" in result
        assert "r_squared" in result
        assert isinstance(result["coefficients"]["age"], float)

    def test_rejects_target_in_features(self, dummy_csv_path: Path) -> None:
        """Inject a request where `age` is both the target and a feature.

        The tool must explicitly reject target leakage to prevent infinite or
        trivial matrix inversions.
        """
        with pytest.raises(ValueError, match="Target column 'age' cannot also appear in features."):
            regression(dummy_csv_path, target="age", features=["age", "fare"])

    def test_rejects_insufficient_rows_after_dropna(self, dummy_csv_path: Path) -> None:
        """Inject a request with 2 features on a dataset with only 3 complete rows.

        The tool must detect the lack of degrees of freedom (`features + 2` required)
        and raise a ValueError preventing singular matrix errors.
        """
        with pytest.raises(
            ValueError, match="Insufficient complete rows \\(3\\) to fit regression with 2 feature\\(s\\)."
        ):
            regression(dummy_csv_path, target="fare", features=["age", "passenger_id"])

    def test_rejects_hallucinated_column(self, dummy_csv_path: Path) -> None:
        """Inject a non-existent target column 'ghost'.

        The tool must reject the hallucinated column before attempting to fit the model.
        """
        with pytest.raises(ValueError, match="Column 'ghost' not found."):
            regression(dummy_csv_path, target="ghost", features=["age"])


class TestDetectOutliers:
    """Tests validating the deterministic outlier detection engine."""

    def test_use_case_executes_iqr_outliers(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for IQR outlier detection on `fare`.

        Asserts the tool computes the bounds and returns the exact schema.
        """
        result = detect_outliers(dummy_csv_path, column="fare", method="iqr")

        assert result["method"] == "iqr"
        assert result["column"] == "fare"
        assert "outlier_count" in result
        assert "outlier_examples" in result
        assert "params" in result
        assert "lower_bound" in result["params"]

    def test_use_case_executes_zscore_outliers(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for Z-score outlier detection on `age`.

        Asserts the tool computes the mean/std and returns the exact schema.
        """
        result = detect_outliers(dummy_csv_path, column="age", method="z_score")

        assert result["method"] == "z_score"
        assert "mean" in result["params"]
        assert "std" in result["params"]

    def test_rejects_zero_variance_zscore(self, constant_column_csv: Path) -> None:
        """Inject a request for Z-score outlier detection on a constant column.

        The tool must detect the zero standard deviation and raise a ValueError
        preventing a division-by-zero error during Z-score calculation.
        """
        with pytest.raises(
            ValueError, match="Cannot compute z-score outliers for 'constant_val': zero or undefined variance."
        ):
            detect_outliers(constant_column_csv, column="constant_val", method="z_score")

    def test_rejects_non_numeric_column(self, dummy_csv_path: Path) -> None:
        """Inject a string column ('embarked') into the outlier request.

        The tool must enforce numeric dtypes to prevent Pandas quantile errors.
        """
        with pytest.raises(ValueError, match="Column 'embarked' is not numeric"):
            detect_outliers(dummy_csv_path, column="embarked", method="iqr")

    def test_rejects_hallucinated_method(self, dummy_csv_path: Path) -> None:
        """Inject an unsupported method like 'dbscan'.

        The tool must reject the hallucinated method and list valid options.
        """
        with pytest.raises(ValueError, match="Unsupported outlier method 'dbscan'. Choose from:") as exc_info:
            detect_outliers(dummy_csv_path, column="age", method="dbscan")

        assert "'iqr'" in str(exc_info.value)


class TestRequestColumnStats:
    """Tests validating the lazy-fetch column profiling tool."""

    def test_use_case_executes_lazy_fetch(self, dummy_csv_path: Path) -> None:
        """Inject a valid request for detailed stats on `age` and `fare`.

        Asserts the tool delegates to `compute_detailed_stats` and returns the
        exact nested dictionary structure.
        """
        result = request_column_stats(dummy_csv_path, columns=["age", "fare"])

        assert "detailed_stats" in result
        assert "age" in result["detailed_stats"]
        assert "fare" in result["detailed_stats"]
        assert result["detailed_stats"]["age"]["dtype"] == "float64"

    def test_rejects_empty_columns_list(self, dummy_csv_path: Path) -> None:
        """Inject an empty list of columns.

        The tool must reject the no-op request immediately.
        """
        with pytest.raises(ValueError, match="request_column_stats requires at least 1 column."):
            request_column_stats(dummy_csv_path, columns=[])

    def test_rejects_hallucinated_columns_with_exact_error(self, dummy_csv_path: Path) -> None:
        """Inject a list containing a non-existent column 'ghost'.

        The tool must raise a ValueError exposing the exact missing column
        and the available columns list.
        """
        with pytest.raises(ValueError, match="Column\\(s\\) not found: \\['ghost'\\].") as exc_info:
            request_column_stats(dummy_csv_path, columns=["age", "ghost"])

        assert "Available: ['passenger_id', 'survived', 'age', 'fare', 'embarked']" in str(exc_info.value)

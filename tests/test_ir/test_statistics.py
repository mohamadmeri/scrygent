"""Tests for the statistics IR models and their logical constraints."""

import pytest
from pydantic import ValidationError

# Adjust imports based on your exact file structure
from scrygent.contracts import CorrelationMethod, RegressionMethod, OutlierMethod
from scrygent.ir.statistics import (
    ColumnStatsParams,
    CorrelationParams,
    OutlierParams,
    RegressionParams,
)


class TestCorrelationParams:
    """Verifies boundaries for the CorrelationParams model."""

    def test_valid_correlation_defaults(self):
        """Ensure the LLM can omit the method and it defaults correctly."""
        model = CorrelationParams(**{"columns": ["age", "salary"]}) # type: ignore
        assert model.columns == ["age", "salary"]
        assert model.method == CorrelationMethod.PEARSON

    def test_requires_minimum_two_columns(self):
        """Correlation requires at least two variables."""
        with pytest.raises(ValidationError) as exc_info:
            CorrelationParams(**{"columns": ["age"]}) # type: ignore
        assert "List should have at least 2 items" in str(exc_info.value)

    def test_invalid_correlation_method_rejected(self):
        """Ensure hallucinated enum values are caught."""
        with pytest.raises(ValidationError) as exc_info:
            CorrelationParams(**{
                "columns": ["age", "salary"],
                "method": "MAGIC_MATH"
            })
        assert "Input should be" in str(exc_info.value)


class TestRegressionParams:
    """Verifies boundaries and logic for the RegressionParams model."""

    def test_valid_regression_defaults(self):
        """Ensure valid instantiation with defaults."""
        model = RegressionParams(**{
            "target": "salary",
            "features": ["age", "experience"]
        })
        assert model.target == "salary"
        assert model.features == ["age", "experience"]
        assert model.method == RegressionMethod.LINEAR

    def test_target_cannot_be_in_features(self):
        """The logical guard must prevent the LLM from regressing a target on itself."""
        with pytest.raises(ValidationError) as exc_info:
            RegressionParams(**{
                "target": "salary",
                "features": ["age", "salary"]
            })
        assert "target 'salary' cannot also appear in features" in str(exc_info.value)

    def test_requires_minimum_one_feature(self):
        """Regression requires at least one independent variable."""
        with pytest.raises(ValidationError) as exc_info:
            RegressionParams(**{
                "target": "salary",
                "features": []
            })
        assert "List should have at least 1 item" in str(exc_info.value)


class TestOutlierParams:
    """Verifies boundaries for the OutlierParams model."""

    def test_valid_outlier_defaults(self):
        """Ensure valid instantiation with defaults."""
        model = OutlierParams(**{"column": "salary"}) # type: ignore
        assert model.column == "salary"
        assert model.method == OutlierMethod.IQR


class TestColumnStatsParams:
    """Verifies boundaries for the ColumnStatsParams model."""

    def test_requires_minimum_one_column(self):
        """The LLM cannot request stats for zero columns."""
        with pytest.raises(ValidationError) as exc_info:
            ColumnStatsParams(**{"columns": []})
        assert "List should have at least 1 item" in str(exc_info.value)

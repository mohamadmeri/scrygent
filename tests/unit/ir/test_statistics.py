"""Destructive test suite for the statistics Intermediate Representation models.

This module aggressively tests the Pydantic IR schemas for correlation,
regression, outlier detection, and column statistics. It ensures that
hallucinated methods, logical impossibilities (like target leakage into
features), and boundary-polluting types are strictly rejected before
reaching the deterministic execution engine.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scrygent.contracts.statistics import CorrelationMethod, OutlierMethod
from scrygent.ir.statistics import ColumnStatsParams, CorrelationParams, OutlierParams, RegressionParams


class TestCorrelationParams:
    """Tests validating the strict schema and boundary enforcement of correlation IR."""

    def test_accepts_valid_payload_and_scrubs_numpy_array_columns(self) -> None:
        """Inject a valid payload containing a NumPy array for the `columns` field.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.ndarray` into a native Python `list` of `str`.
        """
        payload: dict[str, Any] = {
            "columns": np.array(["age", "fare"], dtype=object),
            "method": CorrelationMethod.SPEARMAN,
        }
        model = CorrelationParams(**payload)

        assert model.method == CorrelationMethod.SPEARMAN
        assert isinstance(model.columns, list)
        assert model.columns == ["age", "fare"]

    def test_rejects_single_column_list_with_min_length_error(self) -> None:
        """Inject a list containing only one column.

        Correlation requires at least two columns to compute a matrix. The schema
        enforces `min_length=2` to prevent KeyError or index errors in the engine.
        """
        payload: dict[str, Any] = {"columns": ["age"]}
        with pytest.raises(ValidationError) as exc_info:
            CorrelationParams(**payload)

        assert "List should have at least 2 items" in str(exc_info.value)

    def test_rejects_hallucinated_correlation_method(self) -> None:
        """Inject an unsupported method string like 'covariance'.

        The schema must reject hallucinated methods to prevent attribute errors
        in the Pandas `.corr()` execution.
        """
        payload: dict[str, Any] = {"columns": ["age", "fare"], "method": "covariance"}
        with pytest.raises(ValidationError) as exc_info:
            CorrelationParams(**payload)

        assert "Input should be" in str(exc_info.value)
        assert "'pearson'" in str(exc_info.value)


class TestRegressionParams:
    """Tests validating the strict schema and logical constraints of regression IR."""

    def test_accepts_valid_regression_payload(self) -> None:
        """Verify a baseline valid regression payload passes schema validation."""
        payload: dict[str, Any] = {"target": "survived", "features": ["age", "fare"]}
        model = RegressionParams(**payload)

        assert model.target == "survived"
        assert model.features == ["age", "fare"]

    def test_rejects_target_listed_in_features_with_exact_error(self) -> None:
        """Inject the target column also as a feature.

        The custom model validator must catch this data leakage and raise a
        ValueError preventing the execution engine from training on the label.
        """
        payload: dict[str, Any] = {
            "target": "survived",
            "features": ["age", "survived"],
        }
        with pytest.raises(ValidationError) as exc_info:
            RegressionParams(**payload)

        assert "target 'survived' cannot also appear in features." in str(exc_info.value)

    def test_rejects_empty_features_list(self) -> None:
        """Inject an empty list for the `features` field.

        The schema enforces `min_length=1` because a regression requires at
        least one feature to compute.
        """
        payload: dict[str, Any] = {"target": "survived", "features": []}
        with pytest.raises(ValidationError) as exc_info:
            RegressionParams(**payload)

        assert "List should have at least 1 item" in str(exc_info.value)

    def test_rejects_missing_target_field(self) -> None:
        """Attempt to instantiate the model without the `target` field.

        Ensures strict failure when the LLM drops required parameters.
        """
        payload: dict[str, Any] = {"features": ["age"]}
        with pytest.raises(ValidationError) as exc_info:
            RegressionParams(**payload)

        assert "Field required" in str(exc_info.value)
        assert "target" in str(exc_info.value)


class TestOutlierParams:
    """Tests validating the strict schema and closed vocabulary of outlier IR."""

    def test_accepts_valid_outlier_payload(self) -> None:
        """Verify a baseline valid outlier payload passes schema validation."""
        payload: dict[str, Any] = {"column": "fare", "method": OutlierMethod.Z_SCORE}
        model = OutlierParams(**payload)

        assert model.column == "fare"
        assert model.method == OutlierMethod.Z_SCORE

    def test_rejects_hallucinated_outlier_method(self) -> None:
        """Inject an unsupported method string like 'isolation_forest'.

        The schema must reject hallucinated methods to prevent attribute errors
        in the outlier detection execution.
        """
        payload: dict[str, Any] = {"column": "fare", "method": "isolation_forest"}
        with pytest.raises(ValidationError) as exc_info:
            OutlierParams(**payload)

        assert "Input should be" in str(exc_info.value)
        assert "'iqr'" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `threshold` field.

        The `extra="forbid"` rule must apply to prevent schema drift and
        silent acceptance of unused LLM parameters.
        """
        payload: dict[str, Any] = {
            "column": "fare",
            "method": OutlierMethod.IQR,
            "threshold": 1.5,
        }
        with pytest.raises(ValidationError) as exc_info:
            OutlierParams(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestColumnStatsParams:
    """Tests validating the strict schema for on-demand column statistics fetching."""

    def test_accepts_valid_columns_list(self) -> None:
        """Verify a baseline valid columns list passes schema validation."""
        payload: dict[str, Any] = {"columns": ["age", "fare", "pclass"]}
        model = ColumnStatsParams(**payload)

        assert len(model.columns) == 3

    def test_rejects_empty_columns_list(self) -> None:
        """Inject an empty list for the `columns` field.

        The schema enforces `min_length=1` because requesting stats for zero
        columns is a no-op that wastes execution cycles.
        """
        payload: dict[str, Any] = {"columns": []}
        with pytest.raises(ValidationError) as exc_info:
            ColumnStatsParams(**payload)

        assert "List should have at least 1 item" in str(exc_info.value)

    def test_rejects_non_string_elements_in_columns_list(self) -> None:
        """Inject a list containing integers instead of strings.

        The schema must enforce strict string types for column names to prevent
        implicit type coercion bugs in Pandas indexing.
        """
        payload: dict[str, Any] = {"columns": [1, 2, 3]}  # type: ignore[dict-item]
        with pytest.raises(ValidationError) as exc_info:
            ColumnStatsParams(**payload)

        assert "Input should be a valid string" in str(exc_info.value)

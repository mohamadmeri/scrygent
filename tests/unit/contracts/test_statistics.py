"""Destructive test suite for the statistics tool contracts.

This module ensures the closed-vocabulary enums for correlation, regression,
and outlier detection strictly enforce their allowed values and reject
hallucinated or invalid algorithm identifiers at the boundary.
"""

import pytest

from scrygent.contracts.statistics import CorrelationMethod, OutlierMethod, RegressionMethod


class TestCorrelationMethodContract:
    """Validates the exact closed vocabulary and type strictness of the correlation method enum."""

    def test_correlation_method_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected methods and no others.

        Asserts that the system cannot be extended with new correlation algorithms
        without explicitly modifying this contract.
        """
        assert len(CorrelationMethod) == 3
        assert CorrelationMethod.PEARSON == "pearson"
        assert CorrelationMethod.SPEARMAN == "spearman"
        assert CorrelationMethod.KENDALL == "kendall"

        members = [member.value for member in CorrelationMethod]
        assert set(members) == {"pearson", "spearman", "kendall"}

    def test_correlation_method_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported method string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates an unsupported operation like 'covariance'.
        """
        with pytest.raises(ValueError, match="'covariance' is not a valid CorrelationMethod"):
            CorrelationMethod("covariance")

    def test_correlation_method_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers or objects to prevent implicit type
        coercion bugs in the executor layer.
        """
        with pytest.raises(ValueError):
            CorrelationMethod(1)  # type: ignore[arg-type]


class TestRegressionMethodContract:
    """Validates the exact closed vocabulary and type strictness of the regression method enum."""

    def test_regression_method_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected methods and no others.

        Asserts that the system cannot be extended with new regression algorithms
        without explicitly modifying this contract.
        """
        assert len(RegressionMethod) == 1
        assert RegressionMethod.LINEAR == "linear"

        members = [member.value for member in RegressionMethod]
        assert set(members) == {"linear"}

    def test_regression_method_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported method string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates an unsupported operation like 'logistic'.
        """
        with pytest.raises(ValueError, match="'logistic' is not a valid RegressionMethod"):
            RegressionMethod("logistic")

    def test_regression_method_enum_rejects_attribute_access_for_unknown_methods(self) -> None:
        """Attempt to access a non-existent method via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = RegressionMethod.RIDGE  # type: ignore[attr-defined]


class TestOutlierMethodContract:
    """Validates the exact closed vocabulary and type strictness of the outlier method enum."""

    def test_outlier_method_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected methods and no others.

        Asserts that the system cannot be extended with new outlier algorithms
        without explicitly modifying this contract.
        """
        assert len(OutlierMethod) == 2
        assert OutlierMethod.IQR == "iqr"
        assert OutlierMethod.Z_SCORE == "z_score"

        members = [member.value for member in OutlierMethod]
        assert set(members) == {"iqr", "z_score"}

    def test_outlier_method_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported method string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates an unsupported operation like 'dbscan'.
        """
        with pytest.raises(ValueError, match="'dbscan' is not a valid OutlierMethod"):
            OutlierMethod("dbscan")

    def test_outlier_method_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject None or integers to prevent implicit type
        coercion bugs.
        """
        with pytest.raises(ValueError):
            OutlierMethod(None)  # type: ignore[arg-type]

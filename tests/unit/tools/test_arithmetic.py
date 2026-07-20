"""Destructive and functional test suite for the arithmetic engine.

This module aggressively tests the safe `numexpr` evaluation boundary.
It ensures that unknown identifiers, non-numeric columns, and syntax
errors are caught and formatted for the LLM correction loop, while also
validating actual mathematical use cases end-to-end.
"""

from pathlib import Path

import pytest

from scrygent.tools.arithmetic import derive_column, evaluate_metrics


class TestDeriveColumn:
    """Tests validating the safe row-wise column derivation tool."""

    def test_use_case_executes_valid_arithmetic_and_swaps_csv_path(
        self,
        dummy_csv_path: Path,
    ) -> None:
        """Inject a valid expression (`age + fare`) against the dummy dataset.

        Asserts the tool successfully computes the row-wise math, writes a new
        CSV to disk, and returns the exact statistics of the new column.
        """
        result = derive_column(
            current_csv_path=dummy_csv_path,
            new_column="total_value",
            expression="age + fare",
        )

        assert isinstance(result["current_csv_path"], str)
        new_path = Path(result["current_csv_path"])
        assert new_path.exists()
        assert new_path != dummy_csv_path

        assert result["new_column"] == "total_value"
        assert result["expression"] == "age + fare"

        # 22.0 + 7.25 = 29.25
        assert result["sample"]["min"] == 29.25
        # 38.0 + 71.28 = 109.28
        assert result["sample"]["max"] == 109.28

    def test_use_case_executes_numexpr_builtin_function(
        self,
        dummy_csv_path: Path,
    ) -> None:
        """Inject a valid expression using a `numexpr` function (`sqrt(age)`).

        Asserts the tool recognizes the function, does not treat it as an
        unknown column, and successfully evaluates the math.
        """
        result = derive_column(
            current_csv_path=dummy_csv_path,
            new_column="age_sqrt",
            expression="sqrt(age)",
        )

        assert result["new_column"] == "age_sqrt"
        # sqrt(22.0) = 4.6904
        assert result["sample"]["min"] == pytest.approx(4.6904, rel=1e-4)

    def test_rejects_empty_expression_string(self, dummy_csv_path: Path) -> None:
        """Inject an empty string for the `expression` field.

        The tool must explicitly reject empty expressions before hitting `numexpr`.
        """
        with pytest.raises(ValueError, match="derive_column requires a non-empty expression."):
            derive_column(dummy_csv_path, "new_col", "")

    def test_rejects_existing_column_name(self, dummy_csv_path: Path) -> None:
        """Inject a `new_column` name that already exists in the DataFrame.

        The tool must prevent silent overwrites of original data.
        """
        with pytest.raises(ValueError, match="Column 'age' already exists. Choose a distinct name."):
            derive_column(dummy_csv_path, "age", "fare * 2")

    def test_rejects_expression_with_no_column_references(self, dummy_csv_path: Path) -> None:
        """Inject a static expression like `1 + 1`.

        The tool must enforce that derived columns actually use dataset variables.
        """
        with pytest.raises(ValueError, match="Expression must reference at least 1 existing column."):
            derive_column(dummy_csv_path, "static_col", "1 + 1")

    def test_rejects_hallucinated_column_identifier(self, dummy_csv_path: Path) -> None:
        """Inject an expression referencing a non-existent column `fake_col`.

        The tool must extract the identifier, fail to find it, and raise an
        error listing the available columns for the LLM correction chain.
        """
        with pytest.raises(
            ValueError, match="Expression references unknown identifier\\(s\\): \\['fake_col'\\]."
        ) as exc_info:
            derive_column(dummy_csv_path, "new_col", "age + fake_col")

        assert "Available columns: ['age', 'embarked', 'fare', 'passenger_id', 'survived']" in str(exc_info.value)

    def test_rejects_non_numeric_column_in_expression(self, dummy_csv_path: Path) -> None:
        """Inject an expression referencing a string column (`embarked`).

        The tool must validate dtypes before passing arrays to `numexpr`,
        preventing Pandas/NumPy casting exceptions.
        """
        with pytest.raises(ValueError, match="Expression references non-numeric column\\(s\\): \\['embarked'\\]"):
            derive_column(dummy_csv_path, "new_col", "age + embarked")

    def test_rejects_malformed_expression_syntax(self, dummy_csv_path: Path) -> None:
        """Inject a syntactically invalid expression (`age +`).

        The tool must catch the `numexpr` exception, suppress the verbose
        traceback, and raise a concise ValueError.
        """
        with pytest.raises(ValueError, match="Failed to evaluate expression 'age \\+':"):
            derive_column(dummy_csv_path, "new_col", "age +")


class TestEvaluateMetrics:
    """Tests validating the safe scalar metric evaluation tool."""

    def test_use_case_executes_valid_scalar_math(self) -> None:
        """Inject a valid expression (`revenue / cost`) with a values dict.

        Asserts the tool evaluates the math correctly and returns the exact float.
        """
        result = evaluate_metrics(
            expression="revenue / cost",
            values={"revenue": 1000.0, "cost": 4.0},
        )

        assert result["expression"] == "revenue / cost"
        assert isinstance(result["result"], float)
        assert result["result"] == 250.0

    def test_use_case_executes_with_mixed_case_and_functions(self) -> None:
        """Inject an expression using `abs()` and mixed case variables.

        Asserts `numexpr` handles the function and maps the variables correctly.
        """
        result = evaluate_metrics(
            expression="abs(Net_Loss)",
            values={"Net_Loss": -42.5},
        )

        assert result["result"] == 42.5

    def test_rejects_empty_expression_string(self) -> None:
        """Inject an empty string for the `expression` field.

        The tool must explicitly reject empty expressions.
        """
        with pytest.raises(ValueError, match="evaluate_metrics requires a non-empty expression."):
            evaluate_metrics("", {"a": 1.0})

    def test_rejects_empty_values_dictionary(self) -> None:
        """Inject an empty dictionary for the `values` field.

        The tool must enforce at least one scalar input.
        """
        with pytest.raises(ValueError, match="evaluate_metrics requires at least 1 named value."):
            evaluate_metrics("1 + 1", {})

    def test_rejects_expression_with_no_provided_value_refs(self) -> None:
        """Inject a static expression like `2 * 2` alongside a values dict.

        The tool must enforce that the expression actually uses the provided variables.
        """
        with pytest.raises(ValueError, match="Expression must reference at least 1 provided value."):
            evaluate_metrics("2 * 2", {"a": 1.0})

    def test_rejects_hallucinated_value_identifier(self) -> None:
        """Inject an expression referencing a variable not in the `values` dict.

        The tool must raise an error exposing the missing identifier.
        """
        with pytest.raises(
            ValueError, match="Expression references unknown identifier\\(s\\): \\['tax'\\]."
        ) as exc_info:
            evaluate_metrics("revenue - tax", {"revenue": 100.0})

        assert "Available columns: ['revenue']" in str(exc_info.value)

    def test_rejects_malformed_expression_syntax(self) -> None:
        """Inject a syntactically invalid expression (`revenue /`).

        The tool must catch the `numexpr` exception and raise a concise ValueError.
        """
        with pytest.raises(ValueError, match="Failed to evaluate expression 'revenue /':"):
            evaluate_metrics("revenue /", {"revenue": 100.0})

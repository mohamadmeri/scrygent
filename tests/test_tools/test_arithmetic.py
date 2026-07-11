"""Tests for arithmetic tools: derive_column and evaluate_metrics."""
import pandas as pd
import pytest
from scrygent.tools.arithmetic import derive_column, evaluate_metrics


# ── Fixtures ──
@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV with numeric columns for derivation."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0],
        "cost": [80.0, 150.0, 250.0],
        "units": [10, 20, 30],
        "category": ["A", "B", "A"],  # non-numeric
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def empty_csv(tmp_path):
    """Empty CSV for edge cases."""
    df = pd.DataFrame()
    csv_path = tmp_path / "empty.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ── derive_column ──
class TestDeriveColumn:
    def test_basic_expression(self, sample_csv):
        result = derive_column(
            current_csv_path=sample_csv,
            new_column="profit",
            expression="revenue - cost",
        )
        assert "current_csv_path" in result
        assert result["new_column"] == "profit"
        assert result["expression"] == "revenue - cost"
        # check sample stats roughly
        assert result["sample"]["min"] == pytest.approx(20.0)
        assert result["sample"]["max"] == pytest.approx(50.0)
        assert result["sample"]["mean"] == pytest.approx(40.0)

    def test_with_numexpr_function(self, sample_csv):
        result = derive_column(
            current_csv_path=sample_csv,
            new_column="sqrt_rev",
            expression="sqrt(revenue)",
        )
        assert "current_csv_path" in result
        # The new CSV should have the column
        df = pd.read_csv(result["current_csv_path"])
        assert "sqrt_rev" in df.columns
        assert df["sqrt_rev"].iloc[0] == pytest.approx(10.0)

    def test_multiple_columns(self, sample_csv):
        result = derive_column(
            current_csv_path=sample_csv,
            new_column="profit_per_unit",
            expression="(revenue - cost) / units",
        )
        df = pd.read_csv(result["current_csv_path"])
        assert df["profit_per_unit"].iloc[0] == pytest.approx(2.0)

    def test_unknown_identifier_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Expression references unknown identifier"):
            derive_column(
                current_csv_path=sample_csv,
                new_column="x",
                expression="revenue + unknown_col",
            )

    def test_non_numeric_column_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Expression references non-numeric column"):
            derive_column(
                current_csv_path=sample_csv,
                new_column="x",
                expression="category * 2",
            )

    def test_duplicate_column_name_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Column 'revenue' already exists"):
            derive_column(
                current_csv_path=sample_csv,
                new_column="revenue",
                expression="revenue * 1",
            )

    def test_empty_expression_raises(self, sample_csv):
        with pytest.raises(ValueError, match="non-empty expression"):
            derive_column(
                current_csv_path=sample_csv,
                new_column="x",
                expression="   ",
            )

    def test_expression_with_no_column_refs_raises(self, sample_csv):
        with pytest.raises(ValueError, match="at least 1 existing column"):
            derive_column(
                current_csv_path=sample_csv,
                new_column="x",
                expression="42",
            )

    def test_invalid_expression_syntax_raises(self, sample_csv):
        with pytest.raises(ValueError, match="Failed to evaluate expression"):
            derive_column(
                current_csv_path=sample_csv,
                new_column="x",
                expression="revenue + (",
            )

    def test_nonexistent_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            derive_column(
                current_csv_path="/nonexistent/file.csv",
                new_column="x",
                expression="a + b",
            )

    def test_output_csv_has_original_and_new_column(self, sample_csv):
        result = derive_column(
            current_csv_path=sample_csv,
            new_column="profit",
            expression="revenue - cost",
        )
        df = pd.read_csv(result["current_csv_path"])
        assert "revenue" in df.columns
        assert "cost" in df.columns
        assert "profit" in df.columns
        assert len(df) == 3


# ── evaluate_metrics ──
class TestEvaluateMetrics:
    def test_basic_ratio(self):
        result = evaluate_metrics(
            expression="a / b",
            values={"a": 10.0, "b": 2.0},
        )
        assert result["expression"] == "a / b"
        assert result["result"] == 5.0

    def test_complex_expression(self):
        result = evaluate_metrics(
            expression="(a + b) * c",
            values={"a": 2, "b": 3, "c": 4},
        )
        assert result["result"] == 20.0

    def test_single_value(self):
        result = evaluate_metrics(
            expression="x * 2",
            values={"x": 5.0},
        )
        assert result["result"] == 10.0

    def test_unknown_identifier_raises(self):
        with pytest.raises(ValueError, match="Expression references unknown identifier"):
            evaluate_metrics(
                expression="a + unknown",
                values={"a": 1.0},
            )

    def test_expression_without_provided_value_raises(self):
        with pytest.raises(ValueError, match="must reference at least 1 provided value"):
            evaluate_metrics(expression="2+2", values={"a": 1.0})

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="non-empty expression"):
            evaluate_metrics(
                expression="",
                values={"a": 1.0},
            )

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="at least 1 named value"):
            evaluate_metrics(
                expression="x",
                values={},
            )

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError, match="Failed to evaluate expression"):
            evaluate_metrics(
                expression="a + (",
                values={"a": 1.0},
            )

    def test_result_is_float(self):
        result = evaluate_metrics(expression="x", values={"x": 42.0})
        assert isinstance(result["result"], float)
        assert result["result"] == 42.0
        
    def test_numexpr_function(self):
        result = evaluate_metrics(
            expression="sqrt(a) + log(b)",
            values={"a": 16.0, "b": 1.0},
        )
        assert result["result"] == pytest.approx(4.0)  # log(1)=0

    def test_expression_with_only_numexpr_funcs_and_no_user_columns_raises(self):
        # only numexpr built-in, no user value referenced
        with pytest.raises(ValueError, match="at least 1 provided value"):
            evaluate_metrics(
                expression="sqrt(4)",
                values={"a": 1.0},  # not referenced
            )

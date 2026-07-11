"""Tests for the new profiler – public API and internal column scoring."""
import pandas as pd
from scrygent.tools.profiler import profile_dataframe, _select_columns
from scrygent.tools.profiler import _is_identifier

# ── Helpers ──
def make_df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


# ── Public API (profile_dataframe) ──
class TestProfileDataframe:
    def test_returns_expected_keys(self):
        df = make_df({"a": [1, 2], "b": [3, 4]})
        result = profile_dataframe(df, "a")
        expected = {"row_count", "global_schema", "detailed_stats", "truncated",
                    "row_sample", "missing_detailed_stats"}
        assert set(result.keys()) == expected

    def test_row_count(self):
        df = make_df({"x": range(10)})
        result = profile_dataframe(df, "x")
        assert result["row_count"] == 10

    def test_global_schema_has_all_columns(self):
        df = make_df({"col1": [1], "col2": ["text"]})
        result = profile_dataframe(df, "any")
        assert set(result["global_schema"].keys()) == {"col1", "col2"}
        for v in result["global_schema"].values():
            assert isinstance(v, str)

    def test_row_sample_length_and_nan(self):
        df = make_df({"val": [1.0, None, 3.0]})
        result = profile_dataframe(df, "val")
        assert len(result["row_sample"]) == 3
        # Second row should be None (NaN → null)
        assert result["row_sample"][1]["val"] is None

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = profile_dataframe(df, "query")
        assert result["row_count"] == 0
        assert result["global_schema"] == {}
        assert result["detailed_stats"] == {}

    def test_truncation_flag(self):
        # Create many columns beyond MAX_DETAILED_COLUMNS=15
        df = make_df({f"col_{i}": [1, 2] for i in range(20)})
        result = profile_dataframe(df, "any")
        assert result["truncated"] is True
        assert len(result["detailed_stats"]) <= 15

    def test_missing_detailed_stats_calculated(self):
        df = make_df({f"col_{i}": [1] for i in range(20)})
        result = profile_dataframe(df, "col_0")
        all_cols = set(result["global_schema"].keys())
        stat_cols = set(result["detailed_stats"].keys())
        missing = result["missing_detailed_stats"]
        assert missing == sorted(all_cols - stat_cols)

    def test_integer_column_names_normalized(self):
        df = make_df({0: [1, 2], 1: ["a", "b"]})
        result = profile_dataframe(df, "0")
        # Keys in global_schema should be strings
        assert isinstance(list(result["global_schema"].keys())[0], str)
        assert "0" in result["global_schema"]

    def test_row_sample_keys_are_strings(self):
        df = make_df({0: [1, 2], "text": ["a", "b"]})
        result = profile_dataframe(df, "query")
        sample = result["row_sample"][0]
        for k in sample:
            assert isinstance(k, str)


# ── Internal column selector (_select_columns) ──
class TestSelectColumns:
    def test_direct_query_match_boosted(self):
        df = make_df({"revenue": [1, 2], "cost": [3, 4], "other": [5, 6]})
        cols = _select_columns(df, "revenue", max_cols=2)
        # revenue should appear first due to direct match boost
        assert cols[0] == "revenue"

    def test_identifier_columns_penalized(self):
        df = make_df({"id": [1, 2], "value": [10, 20]})
        cols = _select_columns(df, "value", max_cols=2)
        # value should come before id, because id is an identifier
        assert cols[0] == "value"

    def test_uuid_like_column_penalized(self):
        df = make_df({"uuid": ["a", "b"], "name": ["x", "y"]})
        cols = _select_columns(df, "name", max_cols=2)
        assert cols[0] == "name"

    def test_high_unique_ratio_looks_like_identifier(self):
        df = make_df({"some_id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
        cols = _select_columns(df, "value", max_cols=2)
        assert cols[0] == "value"

    def test_monotonic_increasing_integer_is_penalized(self):
        df = make_df({"row_number": [1, 2, 3, 4], "amount": [10, 20, 30, 40]})
        cols = _select_columns(df, "amount", max_cols=2)
        assert cols[0] == "amount"

    def test_density_boosts_column(self):
        # null_col has lots of NaNs, so dense_col should rank higher
        df = make_df({"dense_col": [1, 2, 3, 4], "sparse_col": [1, None, None, None]})
        cols = _select_columns(df, "any", max_cols=2)
        assert cols[0] == "dense_col"

    def test_numeric_intent_boost(self):
        df = make_df({"profit": [1, 2], "age": [30, 40]})
        # query with $ gives numeric boost
        cols = _select_columns(df, "profit greater than $50", max_cols=2)
        # profit and age are both numeric, profit matches query, so it should be first
        assert cols[0] == "profit"

    def test_stable_ordering_on_ties(self):
        df = make_df({"a": [1, 2], "b": [3, 4]})
        cols = _select_columns(df, "no_match", max_cols=2)
        # no query match, same density; order should follow original column order
        assert cols == ["a", "b"]

    def test_max_cols_truncation(self):
        df = make_df({f"col_{i}": [1, 2] for i in range(10)})
        cols = _select_columns(df, "col_0", max_cols=3)
        assert len(cols) == 3
        assert cols[0] == "col_0"


class TestIsIdentifier:
    def test_small_dataset_nonid_returns_false(self):
        """With fewer than MIN_ROWS_FOR_STATISTICAL_ID_SIGNAL rows, non‑id names return False."""
        series = pd.Series([1, 2, 3])  # 3 rows, not matching id patterns
        assert _is_identifier("value", series) is False

    def test_large_dataset_high_uniqueness_returns_true(self):
        """With 20+ rows and unique_ratio > 0.97, statistical signal triggers."""
        # 25 rows, all unique → ratio 1.0
        series = pd.Series(range(25))
        # Name is not in _ID_PATTERNS, so only statistical path can mark it as identifier
        assert _is_identifier("row_id_like", series) is True

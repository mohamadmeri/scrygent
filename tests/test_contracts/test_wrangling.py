"""Unit tests for the NormalizeMethod StrEnum (contract layer only)."""

import pytest
from src.scrygent.contracts.wrangling import NormalizeMethod


class TestNormalizeMethodMembership:
    EXPECTED = frozenset({
        "MIN_MAX", "Z_SCORE", "LOG", "STRIP", "LOWERCASE", "UPPERCASE", "TITLE_CASE",
    })
    VALUES = {
        "MIN_MAX": "min_max",
        "Z_SCORE": "z_score",
        "LOG": "log",
        "STRIP": "strip",
        "LOWERCASE": "lowercase",
        "UPPERCASE": "uppercase",
        "TITLE_CASE": "title_case",
    }

    def test_member_names(self):
        assert {m.name for m in NormalizeMethod} == self.EXPECTED

    def test_no_duplicate_values(self):
        vals = [m.value for m in NormalizeMethod]
        assert len(vals) == len(set(vals))

    def test_values_match_spec(self):
        for m in NormalizeMethod:
            assert m.value == self.VALUES[m.name]

    def test_values_are_lowercase(self):
        for m in NormalizeMethod:
            assert isinstance(m.value, str)
            assert m.value.islower()
            assert m.value == m.name.lower()


class TestNormalizeMethodCoercion:
    @pytest.mark.parametrize("val", [m.value for m in NormalizeMethod])
    def test_valid_construct(self, val):
        m = NormalizeMethod(val)
        assert m.value == val
        assert m.name == val.upper()

    @pytest.mark.parametrize("invalid", [
        "MIN_MAX",           # uppercase name
        "Min_Max",           # mixed case
        " min_max",          # leading space
        "min_max ",          # trailing space
        "",                  # empty
        "nonexistent",
        "z-score",           # hyphen
        "log10",             # variation
        "strip ",            # trailing space
    ])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError):
            NormalizeMethod(invalid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            NormalizeMethod(None)  # type: ignore[arg-type]

    def test_number_raises(self):
        with pytest.raises(ValueError):
            NormalizeMethod(5)  # type: ignore[arg-type]


class TestNormalizeMethodProtocols:
    def test_iteration_order(self):
        expected_order = [
            "MIN_MAX", "Z_SCORE", "LOG", "STRIP", "LOWERCASE", "UPPERCASE", "TITLE_CASE",
        ]
        assert [m.name for m in NormalizeMethod] == expected_order

    def test_hashable(self):
        s = set(NormalizeMethod)
        assert len(s) == len(NormalizeMethod)

    def test_identity(self):
        for m in NormalizeMethod:
            assert NormalizeMethod(m.value) is m

    def test_bool(self):
        for m in NormalizeMethod:
            assert bool(m)

    def test_repr_and_str(self):
        for m in NormalizeMethod:
            assert m.name in repr(m)
            assert str(m) == m.value


class TestNormalizeMethodSemantics:
    def test_numeric_methods_exist(self):
        assert NormalizeMethod.MIN_MAX.value == "min_max"
        assert NormalizeMethod.Z_SCORE.value == "z_score"
        assert NormalizeMethod.LOG.value == "log"

    def test_string_methods_exist(self):
        assert NormalizeMethod.STRIP.value == "strip"
        assert NormalizeMethod.LOWERCASE.value == "lowercase"
        assert NormalizeMethod.UPPERCASE.value == "uppercase"
        assert NormalizeMethod.TITLE_CASE.value == "title_case"

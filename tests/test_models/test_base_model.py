from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scrygent.base_model import SanitizationError, ScrygentBaseModel, _recursive_sanitize, _sanitize_scalar

class PayloadModel(ScrygentBaseModel):
    payload: dict[Any, Any]

class StrictFloatModel(ScrygentBaseModel):
    value: float

class OutputsModel(ScrygentBaseModel):
    outputs: dict[str, Any]

class TestSanitizeScalar:
    def test_pd_nat_returns_none(self):
        assert _sanitize_scalar(pd.NaT) is None

    def test_pd_timestamp_nat_returns_none(self):
        assert _sanitize_scalar(pd.Timestamp("NaT")) is None

    def test_pd_timestamp_valid_isoformat(self):
        out = _sanitize_scalar(pd.Timestamp("2026-01-01T12:30:00"))
        assert isinstance(out, str) and out.startswith("2026-01-01")

    def test_np_datetime64_valid_and_nat(self):
        assert _sanitize_scalar(np.datetime64("2026-01-01T12:30:00")).startswith("2026-01-01")
        assert _sanitize_scalar(np.datetime64("NaT")) is None

    def test_python_datetime(self):
        assert _sanitize_scalar(datetime(2026, 7, 9, 14, 12, 10)) == "2026-07-09T14:12:10"

    def test_np_str_int_float_bool(self):
        assert type(_sanitize_scalar(np.str_("hi"))) is str
        assert type(_sanitize_scalar(np.int64(42))) is int
        assert type(_sanitize_scalar(np.float64(0.99))) is float
        assert type(_sanitize_scalar(np.bool_(True))) is bool

    def test_nan_inf_become_none(self):
        assert _sanitize_scalar(np.float64("nan")) is None
        assert _sanitize_scalar(float("nan")) is None
        assert _sanitize_scalar(np.inf) is None
        assert _sanitize_scalar(float("inf")) is None

    def test_none_str_int_bool_passthrough(self):
        assert _sanitize_scalar(None) is None
        assert _sanitize_scalar("x") == "x"
        assert _sanitize_scalar(7) == 7
        assert _sanitize_scalar(True) is True

    def test_pd_na_becomes_none(self):
        assert _sanitize_scalar(pd.NA) is None

    def test_unsupported_raises(self):
        with pytest.raises(SanitizationError, match="has no sanitization rule"):
            _sanitize_scalar(frozenset([1]))

class TestRecursiveSanitize:
    def test_dataframe_and_series_rejected(self):
        with pytest.raises(SanitizationError, match="DataFrame objects cannot cross"):
            _recursive_sanitize(pd.DataFrame())
        with pytest.raises(SanitizationError, match="Series objects cannot cross"):
            _recursive_sanitize(pd.Series([1]))

    def test_dict_key_and_value_sanitized(self):
        clean = _recursive_sanitize({np.int64(1): np.float64(2.5)})
        assert list(clean.keys())[0] == 1 and type(list(clean.keys())[0]) is int
        assert clean[1] == 2.5

    def test_list_tuple_ndarray(self):
        assert _recursive_sanitize([np.int64(1), np.nan]) == [1, None]
        assert _recursive_sanitize((np.str_("a"), pd.NaT)) == ["a", None]
        assert _recursive_sanitize(np.array([1, 2, 3], dtype=np.int64)) == [1, 2, 3]

    def test_nested_fully_sanitized(self):
        dirty = {"a": [{"b": np.bool_(True), "c": pd.Timestamp("2026-01-01")}]}
        clean = _recursive_sanitize(dirty)
        assert clean["a"][0]["b"] is True and isinstance(clean["a"][0]["c"], str)

class TestScrygentBaseModelBoundary:
    def test_extra_forbidden(self):
        class S(ScrygentBaseModel):
            a: int
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            S(a=1, b=2) # type: ignore

    def test_validate_assignment_sanitizes_mutation(self):
        m = StrictFloatModel(value=1.0)
        m.value = np.int64(5) # type: ignore
        assert type(m.value) is float and m.value == 5.0

    def test_validate_assignment_nan_fails(self):
        m = StrictFloatModel(value=1.0)
        with pytest.raises(ValidationError):
            m.value = np.nan # type: ignore

    def test_nan_inf_create_fails_strict_float(self):
        with pytest.raises(ValidationError):
            StrictFloatModel(value=np.nan) # type: ignore
        with pytest.raises(ValidationError):
            StrictFloatModel(value=np.inf) # type: ignore

    def test_dirty_executor_output_sanitized(self):
        m = OutputsModel(outputs={"s": {"metric": np.float64(0.99), "flag": np.bool_(True), "miss": np.nan}})
        assert type(m.outputs["s"]["metric"]) is float
        assert type(m.outputs["s"]["flag"]) is bool
        assert m.outputs["s"]["miss"] is None

    def test_model_dump_json_safe(self):
        m = PayloadModel(payload={np.int64(42): np.str_("hi")})
        assert '"42"' in m.model_dump_json()

    def test_unexpected_exception_wrapped(self):
        with patch("scrygent.base_model._recursive_sanitize", side_effect=RuntimeError("boom")):
            with pytest.raises(ValidationError, match="boom"):
                PayloadModel(payload={"x": 1})

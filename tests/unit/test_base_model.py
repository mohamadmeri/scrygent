"""Destructive test suite for the JSON Boundary (`ScrygentBaseModel`).

This module aggressively tests the sanitization layer to guarantee that no
non-JSON-native types (Pandas C-types, NumPy scalars, NaNs, Infs) can cross
into the LangGraph state, and that complex Pandas objects are explicitly
rejected to enforce the stateless-tool contract.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import _BoundaryProbe


class TestBoundarySanitization:
    """Tests focused on the recursive scrubbing of valid but non-native types."""

    def test_scrubs_exact_numpy_pandas_payload_to_native_primitives(
        self,
        boundary_probe_model: type[_BoundaryProbe],
        numpy_polluted_payload: dict[str, Any],
        sanitized_counterpart: dict[str, Any],
    ) -> None:
        """Simulate a tool returning a deeply nested dict of NumPy/Pandas C-types.

        Asserts that the `@model_validator(mode="wrap")` successfully intercepts
        the payload and converts every value to its exact native Python primitive
        before Pydantic freezes the model.
        """
        model = boundary_probe_model(payload=numpy_polluted_payload)

        assert model.payload == sanitized_counterpart
        assert isinstance(model.payload["count"], int)
        assert isinstance(model.payload["ratio"], float)
        assert isinstance(model.payload["is_valid"], bool)
        assert model.payload["missing_time"] is None
        assert isinstance(model.payload["timestamp"], str)

    def test_scrubs_nested_numpy_arrays_and_preserves_lists(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Inject a raw `np.ndarray` containing mixed types and NaNs.

        The sanitizer must call `.tolist()` on the array and recursively scrub
        its elements, converting `np.nan` to `None` and `np.int64` to `int`.
        """
        polluted_array = np.array([1, 2.5, np.nan, np.inf], dtype=object)
        payload = {"matrix": polluted_array}

        model = boundary_probe_model(payload=payload)

        assert model.payload["matrix"] == [1, 2.5, None, None]
        assert isinstance(model.payload["matrix"][0], int)
        assert isinstance(model.payload["matrix"][1], float)

    def test_scrubs_path_and_datetime_objects_to_strings(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Inject `pathlib.Path` and `datetime.datetime` objects.

        Paths must become strings, and datetimes must become ISO 8601 strings.
        """
        dt = datetime(2024, 1, 1, 12, 0, 0)
        p = Path("/tmp/scrygent/test.csv")
        payload = {"path": p, "timestamp": dt}

        model = boundary_probe_model(payload=payload)

        assert model.payload["path"] == "/tmp/scrygent/test.csv"
        assert isinstance(model.payload["path"], str)
        assert model.payload["timestamp"] == "2024-01-01T12:00:00"
        assert isinstance(model.payload["timestamp"], str)


class TestBoundaryRejection:
    """Tests focused on the strict rejection of unsanitizable or leakable objects."""

    def test_rejects_pandas_dataframe_with_exact_error(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Attempt to pass a raw `pd.DataFrame` into the model.

        The sanitizer must explicitly reject DataFrames to enforce the
        stateless-tool contract, raising a validation error containing the
        precise rejection message rather than attempting naive serialization.
        """
        df = pd.DataFrame({"x": [1, 2]})
        payload = {"leaked_frame": df}

        with pytest.raises(ValidationError) as exc_info:
            boundary_probe_model(payload=payload)

        assert "Pandas DataFrame and Series objects cannot cross the Scrygent model boundary." in str(exc_info.value)

    def test_rejects_pandas_series_with_exact_error(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Attempt to pass a raw `pd.Series` into the model.

        Similar to DataFrames, Series objects must be rejected to prevent
        state bloat and implicit statefulness between tools.
        """
        series = pd.Series([1, 2, 3], name="y")
        payload = {"leaked_series": series}

        with pytest.raises(ValidationError) as exc_info:
            boundary_probe_model(payload=payload)

        assert "Pandas DataFrame and Series objects cannot cross the Scrygent model boundary." in str(exc_info.value)

    def test_rejects_custom_unsanitizable_objects(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Attempt to pass a custom Python class instance with no sanitization rule.

        The fallback `pd.isna()` check will fail on arbitrary objects. The
        sanitizer must raise an error containing the specific type name rather
        than letting the object slip through to Pydantic's JSON encoder.
        """

        class SecretObject:
            pass

        payload = {"custom": SecretObject()}

        with pytest.raises(ValidationError) as exc_info:
            boundary_probe_model(payload=payload)

        assert "Value of type 'SecretObject'" in str(exc_info.value)
        assert "has no sanitization rule." in str(exc_info.value)

    def test_rejects_invalid_dictionary_keys(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Attempt to pass a dictionary with a `tuple` as a key.

        Dictionaries are JSON objects, meaning keys must be strings (or
        primitives cast to strings). Tuples are unhashable by JSON standards
        and must be rejected explicitly.
        """
        payload = {("invalid", "key"): "value"}  # type: ignore[dict-item]

        with pytest.raises(ValidationError) as exc_info:
            boundary_probe_model(payload=payload)  # type: ignore[arg-type]

        assert "Value of type 'tuple'" in str(exc_info.value)
        assert "has no sanitization rule." in str(exc_info.value)


class TestPydanticIntegration:
    """Tests ensuring that Pydantic's native validation still functions after sanitization."""

    def test_extra_fields_are_still_forbidden_post_sanitization(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Inject a valid payload alongside an unexpected field.

        The sanitizer should successfully clean the valid fields, but Pydantic's
        `extra="forbid"` configuration must still trigger a `ValidationError`
        for the unknown field. This proves the `mode="wrap"` validator doesn't
        swallow downstream Pydantic checks.
        """
        payload = {"payload": {"a": 1}, "malicious_field": "should fail"}

        with pytest.raises(ValidationError) as exc_info:
            boundary_probe_model(**payload)

        assert "malicious_field" in str(exc_info.value)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_native_types_pass_through_unchanged(
        self,
        boundary_probe_model: type[_BoundaryProbe],
    ) -> None:
        """Inject a payload containing only native JSON types.

        The sanitizer must not alter or wrap native types. This verifies the
        baseline performance path where no scrubbing is required.
        """
        native_payload = {"count": 1, "name": "test", "active": True, "missing": None}
        model = boundary_probe_model(payload=native_payload)

        assert model.payload == native_payload

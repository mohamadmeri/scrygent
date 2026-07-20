"""Destructive test suite for the deterministic execution engine output schemas.

This module aggressively tests the Pydantic IR schemas for profiler results,
visualization metadata, and final reporting payloads. It ensures that missing
required fields, malformed types, and boundary-polluting Pandas objects are
strictly rejected to prevent state bloat and serialization crashes.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scrygent.models.outputs import AnalysisReport, CSVProfile, DirectAnswer, PlotMetadata


class TestCSVProfile:
    """Tests validating the strict schema and boundary enforcement of dataset profiles."""

    def test_accepts_valid_payload_and_scrubs_numpy_row_count(self) -> None:
        """Inject a valid payload containing a NumPy integer for `row_count`.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.int64` into a native Python `int`.
        """
        payload: dict[str, Any] = {
            "row_count": np.int64(891),
            "global_schema": {"age": "int64", "fare": "float64"},
        }
        model = CSVProfile(**payload)

        assert model.row_count == 891
        assert isinstance(model.row_count, int)
        assert not isinstance(model.row_count, np.integer)

    def test_rejects_missing_global_schema_field(self) -> None:
        """Attempt to instantiate the model without the `global_schema` field.

        Ensures strict failure when the Profiler omits the critical schema map,
        which is required for the Planner to route queries.
        """
        payload: dict[str, Any] = {"row_count": 100}
        with pytest.raises(ValidationError) as exc_info:
            CSVProfile(**payload)

        assert "Field required" in str(exc_info.value)
        assert "global_schema" in str(exc_info.value)

    def test_rejects_leaked_dataframe_in_row_sample(self) -> None:
        """Inject a raw `pd.DataFrame` into the `row_sample` list.

        The Hermetic JSON Boundary must explicitly reject DataFrames to enforce
        the stateless-tool contract and prevent memory bloat.
        """
        df = pd.DataFrame({"a": [1, 2]})
        payload: dict[str, Any] = {
            "row_count": 2,
            "global_schema": {"a": "int64"},
            "row_sample": [df],  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            CSVProfile(**payload)

        assert "Pandas DataFrame and Series objects cannot cross the Scrygent model boundary." in str(exc_info.value)

    def test_rejects_non_string_keys_in_global_schema(self) -> None:
        """Inject integer keys into the `global_schema` dictionary.

        The schema must enforce string keys for column names to prevent downstream
        Pandas indexing errors.
        """
        payload: dict[str, Any] = {
            "row_count": 100,
            "global_schema": {1: "int64"},  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            CSVProfile(**payload)

        assert "Input should be a valid string" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `memory_usage` field.

        The `extra="forbid"` rule must apply to prevent schema drift.
        """
        payload: dict[str, Any] = {
            "row_count": 100,
            "global_schema": {"a": "int64"},
            "memory_usage": "1MB",
        }
        with pytest.raises(ValidationError) as exc_info:
            CSVProfile(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestPlotMetadata:
    """Tests validating the strict schema for in-memory visualization references."""

    def test_accepts_valid_payload(self) -> None:
        """Verify a baseline valid plot metadata payload passes schema validation."""
        payload: dict[str, Any] = {
            "plotly_json": '{"data": [{"x": [1, 2]}]}',
            "description": "A simple line chart.",
        }
        model = PlotMetadata(**payload)

        assert model.description == "A simple line chart."

    def test_rejects_missing_plotly_json(self) -> None:
        """Attempt to instantiate the model without the `plotly_json` field.

        Ensures strict failure when a tool attempts to save a plot without
        serializing the figure to JSON.
        """
        payload: dict[str, Any] = {"description": "Missing JSON."}
        with pytest.raises(ValidationError) as exc_info:
            PlotMetadata(**payload)

        assert "Field required" in str(exc_info.value)
        assert "plotly_json" in str(exc_info.value)

    def test_rejects_dict_for_plotly_json_field(self) -> None:
        """Inject a dictionary instead of a JSON string for `plotly_json`.

        The schema must enforce string type to ensure the payload is pre-serialized,
        preventing implicit Pydantic JSON encoders from choking on complex Plotly objects.
        """
        payload: dict[str, Any] = {
            "plotly_json": {"data": [{"x": [1, 2]}]},  # type: ignore[dict-item]
            "description": "Dict instead of string.",
        }
        with pytest.raises(ValidationError) as exc_info:
            PlotMetadata(**payload)

        assert "Input should be a valid string" in str(exc_info.value)


class TestAnalysisReport:
    """Tests validating the strict schema for final synthesized reporting."""

    def test_accepts_valid_payload_with_nested_plots(self) -> None:
        """Verify a baseline valid report payload with nested plot metadata passes."""
        payload: dict[str, Any] = {
            "primary_answer": "The average fare was 32.20.",
            "plots": [
                {
                    "plotly_json": '{"data": []}',
                    "description": "Fare distribution.",
                }
            ],
        }
        model = AnalysisReport(**payload)

        assert model.primary_answer == "The average fare was 32.20."
        assert len(model.plots) == 1
        assert isinstance(model.plots[0], PlotMetadata)

    def test_rejects_missing_primary_answer(self) -> None:
        """Attempt to instantiate the model without the `primary_answer` field.

        Ensures strict failure when the Reporter omits the direct answer, which
        is the core deliverable of the execution graph.
        """
        payload: dict[str, Any] = {"additional_insights": ["Nothing else."]}
        with pytest.raises(ValidationError) as exc_info:
            AnalysisReport(**payload)

        assert "Field required" in str(exc_info.value)
        assert "primary_answer" in str(exc_info.value)

    def test_rejects_non_list_additional_insights(self) -> None:
        """Inject a string instead of a list for the `additional_insights` field.

        The schema must enforce the list type to prevent iteration errors in
        the Streamlit UI layer.
        """
        payload: dict[str, Any] = {
            "primary_answer": "42.",
            "additional_insights": "Just a string.",  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationError) as exc_info:
            AnalysisReport(**payload)

        assert "Input should be a valid list" in str(exc_info.value)

    def test_rejects_malformed_plot_in_list(self) -> None:
        """Inject a list containing a plot dictionary missing `description`.

        The nested Pydantic validation must catch the missing required field
        inside the list item and halt execution.
        """
        payload: dict[str, Any] = {
            "primary_answer": "42.",
            "plots": [{"plotly_json": "{}"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            AnalysisReport(**payload)

        assert "Field required" in str(exc_info.value)
        assert "description" in str(exc_info.value)


class TestDirectAnswer:
    """Tests validating the strict schema for benchmark-mode outputs."""

    def test_accepts_valid_payload_and_scrubs_numpy_bool(self) -> None:
        """Inject a valid payload containing a NumPy string for the answer.

        Asserts the model accepts the payload and the Hermetic JSON Boundary
        scrubs the `np.str_` into a native Python `str`.
        """
        payload: dict[str, Any] = {"answer": np.str_("150")}
        model = DirectAnswer(**payload)

        assert model.answer == "150"
        assert isinstance(model.answer, str)
        assert not isinstance(model.answer, np.str_)

    def test_rejects_missing_answer_field(self) -> None:
        """Attempt to instantiate the model without the `answer` field.

        Ensures strict failure when the Reporter drops the core scalar answer
        in benchmark mode.
        """
        payload: dict[str, Any] = {}
        with pytest.raises(ValidationError) as exc_info:
            DirectAnswer(**payload)

        assert "Field required" in str(exc_info.value)
        assert "answer" in str(exc_info.value)

    def test_rejects_extra_fields_in_payload(self) -> None:
        """Inject a valid payload alongside an unexpected `confidence` field.

        The `extra="forbid"` rule must apply to prevent the Reporter from
        hallucinating metadata not supported by the benchmark evaluation harness.
        """
        payload: dict[str, Any] = {"answer": "True", "confidence": 0.95}
        with pytest.raises(ValidationError) as exc_info:
            DirectAnswer(**payload)

        assert "Extra inputs are not permitted" in str(exc_info.value)

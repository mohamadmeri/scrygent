"""Tests for the wrangling IR models and their structural constraints."""

import pytest
from pydantic import ValidationError

from scrygent.contracts import NormalizeMethod, FilterOperator
from scrygent.ir.wrangling import FilterDatasetParams, NormalizeColumnParams, NoParams


class TestFilterDatasetParams:
    """Verifies boundaries for the FilterDatasetParams model."""

    def test_valid_filter_dataset(self):
        """Ensure a valid filter payload parses correctly via the shared FilterCondition."""
        model = FilterDatasetParams(**{
            "filters": [
                {"column": "status", "operator": FilterOperator.EQ.value, "value": "active"}
            ]
        }) # type: ignore
        assert len(model.filters) == 1
        assert model.filters[0].column == "status"

    def test_requires_minimum_one_filter(self):
        """The LLM cannot request to filter a dataset without providing any filters."""
        with pytest.raises(ValidationError) as exc_info:
            FilterDatasetParams(**{"filters": []})
        assert "List should have at least 1 item" in str(exc_info.value)


class TestNormalizeColumnParams:
    """Verifies boundaries for the NormalizeColumnParams model."""

    def test_valid_normalization(self):
        """Ensure valid instantiation with an expected normalization method."""
        # Assuming Z_SCORE is a valid member of your NormalizeMethod enum
        # Replace with a real value from your codebase if different
        method_str = list(NormalizeMethod)[0].value 
        
        model = NormalizeColumnParams(**{
            "column": "revenue",
            "method": method_str
        }) # type: ignore
        assert model.column == "revenue"
        assert isinstance(model.method, NormalizeMethod)

    def test_invalid_normalization_method_rejected(self):
        """Ensure hallucinated enum values are caught."""
        with pytest.raises(ValidationError) as exc_info:
            NormalizeColumnParams(**{
                "column": "revenue",
                "method": "MAGIC_SCALING"
            }) # type: ignore
        assert "Input should be" in str(exc_info.value)


class TestNoParams:
    """Verifies the strict empty-payload requirement for tools like reset_dataset."""

    def test_valid_empty_payload(self):
        """An empty dictionary is the only valid payload."""
        model = NoParams(**{})
        assert isinstance(model, NoParams)

    def test_rejects_hallucinated_arguments(self):
        """
        If the LLM tries to pass arguments to a tool that takes none, 
        the base model's extra='forbid' config (or standard strictness) should catch it.
        """
        with pytest.raises(ValidationError) as exc_info:
            NoParams(**{"dataset_name": "data.csv", "force": True})
            
        error_str = str(exc_info.value)
        # Pydantic raises "Extra inputs are not permitted" when extra='forbid'
        assert "Extra inputs are not permitted" in error_str

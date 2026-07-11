"""Tests for the strict, discriminated FilterCondition IR."""

import pytest
from pydantic import TypeAdapter, ValidationError

from scrygent.contracts import FilterOperator
from scrygent.ir.filtering import (
    FilterCondition,
    ListFilterCondition,
    ScalarFilterCondition,
    StringFilterCondition,
)

# TypeAdapter allows us to validate an Annotated Union directly 
# without needing a dummy parent BaseModel.
filter_adapter = TypeAdapter(FilterCondition)


class TestFilterDiscriminatorRouting:
    """Verifies the custom discriminator routes payloads to the correct sub-model."""

    def test_routes_to_scalar(self):
        """Scalar operators (e.g., GT) must route to ScalarFilterCondition."""
        payload = {"column": "age", "operator": FilterOperator.GT.value, "value": 18}
        result = filter_adapter.validate_python(payload)
        
        assert isinstance(result, ScalarFilterCondition)
        assert result.value == 18

    def test_routes_to_list(self):
        """List operators (e.g., IN) must route to ListFilterCondition."""
        payload = {"column": "department", "operator": FilterOperator.IN.value, "value": ["HR", "IT"]}
        result = filter_adapter.validate_python(payload)
        
        assert isinstance(result, ListFilterCondition)
        assert result.value == ["HR", "IT"]

    def test_routes_to_string(self):
        """String operators (e.g., CONTAINS) must route to StringFilterCondition."""
        payload = {"column": "name", "operator": FilterOperator.CONTAINS.value, "value": "smith"}
        result = filter_adapter.validate_python(payload)
        
        assert isinstance(result, StringFilterCondition)
        assert result.value == "smith"

    def test_unknown_operator_rejection(self):
        """If the LLM hallucinates an operator, the discriminator should fail it."""
        payload = {"column": "age", "operator": "IS_SIMILAR_TO", "value": 18}
        
        # Change ValidationError to ValueError here:
        with pytest.raises(ValueError) as exc_info:
            filter_adapter.validate_python(payload)
            
        assert "Unrecognized filter operator" in str(exc_info.value)

class TestFilterLogicalTypeGuards:
    """Verifies that the LLM cannot pass invalid value types for a given operator."""

    def test_list_operator_rejects_scalar_value(self):
        """Using 'IN' with a single integer instead of a list must fail."""
        payload = {"column": "id", "operator": FilterOperator.IN.value, "value": 123}
        
        with pytest.raises(ValidationError) as exc_info:
            filter_adapter.validate_python(payload)
            
        # The discriminator correctly routes it to ListFilterCondition, 
        # which then rejects the integer value.
        assert "Input should be a valid list" in str(exc_info.value)

    def test_string_operator_rejects_list_value(self):
        """Using 'CONTAINS' with a list instead of a string must fail."""
        payload = {"column": "name", "operator": FilterOperator.CONTAINS.value, "value": ["a", "b"]}
        
        with pytest.raises(ValidationError) as exc_info:
            filter_adapter.validate_python(payload)
            
        assert "Input should be a valid string" in str(exc_info.value)


class TestFilterBoundaryConstraints:
    """Verifies field-level bounds (empty strings, empty lists)."""

    def test_column_name_cannot_be_empty(self):
        """Across all filter types, the column name must have length >= 1."""
        payload = {"column": "", "operator": FilterOperator.EQ.value, "value": 1}
        
        with pytest.raises(ValidationError) as exc_info:
            filter_adapter.validate_python(payload)
            
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_list_filter_cannot_be_empty(self):
        """An 'IN' clause with an empty list is useless and should be rejected at the boundary."""
        payload = {"column": "id", "operator": FilterOperator.IN.value, "value": []}
        
        with pytest.raises(ValidationError) as exc_info:
            filter_adapter.validate_python(payload)
            
        assert "List should have at least 1 item" in str(exc_info.value)

    def test_string_filter_cannot_be_empty(self):
        """A 'CONTAINS' clause with an empty string matches everything, which is usually an LLM error."""
        payload = {"column": "name", "operator": FilterOperator.CONTAINS.value, "value": ""}
        
        with pytest.raises(ValidationError) as exc_info:
            filter_adapter.validate_python(payload)
            
        assert "String should have at least 1 character" in str(exc_info.value)

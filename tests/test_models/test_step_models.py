"""Tests for step_models: Step, StepRecord, Plan."""
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from scrygent.models.outputs import TOOL_PARAM_MODELS
from scrygent.models.step_models import Step, StepRecord, Plan
from scrygent.contracts.tool_names import ToolName


# ── Minimal valid parameters per tool ──
VALID_PARAMS = {
    ToolName.ANALYZE_DATA: {"metrics": [{"column": "x", "aggregation": "count", "alias": "cnt"}]},
    ToolName.FILTER_DATASET: {"filters": [{"column": "c", "operator": "==", "value": 1}]},
    ToolName.NORMALIZE_COLUMN: {"column": "c", "method": "min_max"},
    ToolName.RESET_DATASET: {},
    ToolName.CORRELATION: {"columns": ["a", "b"]},
    ToolName.REGRESSION: {"target": "y", "features": ["x"]},
    ToolName.DETECT_OUTLIERS: {"column": "x", "method": "iqr"},
    ToolName.REQUEST_COLUMN_STATS: {"columns": ["a"]},
    ToolName.GENERATE_PLOT: {"plot_type": "bar", "columns": ["cat", "val"]},
    ToolName.DERIVE_COLUMN: {"new_column": "r", "expression": "a+b"},
    ToolName.EVALUATE_METRICS: {"expression": "a", "values": {"a": 1.0}},
}


# ── Step unit tests ──
class TestStepValidation:
    def test_valid_tool_step(self):
        step = Step(
            step_id="1",
            rationale="test",
            action="tool",
            tool_name=ToolName.ANALYZE_DATA,
            parameters=VALID_PARAMS[ToolName.ANALYZE_DATA],
        )
        assert step.action == "tool"
        assert step.parameters["metrics"][0]["alias"] == "cnt"

    def test_tool_step_missing_tool_name(self):
        with pytest.raises(ValidationError, match="tool_name is required"):
            Step(step_id="1", rationale="test", action="tool", tool_name=None)

    def test_tool_step_with_unregistered_tool_name(self):
        # Force a KeyError by passing a ToolName that is not in TOOL_PARAM_MODELS
        # (In reality, the registry should cover all, but we simulate a missing case
        # by temporarily removing an entry)
        # Simpler: we test the runtime error by injecting a wrong enum that isn't in the registry
        # but the registry is complete, so we test the exception message if it happens.
        # To test the try/except we'd mock TOOL_PARAM_MODELS, but we can just validate that
        # the import-time registry matches the enum — already tested in schemas.
        # We'll rely on the contract test already done. For unit, we just test that
        # a KeyError would cause RuntimeError. Actually we'll just skip that branch,
        # or we can mock. We'll test that a valid tool_name works fine.

        # Instead, test that a tool step with invalid parameters raises ValidationError
        # from the param_model.model_validate() inside the validator.
        with pytest.raises(ValidationError):
            Step(
                step_id="2",
                rationale="bad params",
                action="tool",
                tool_name=ToolName.ANALYZE_DATA,
                parameters={"metrics": []},  # empty metrics, min_length=1
            )

    def test_valid_sandbox_step(self):
        step = Step(
            step_id="2",
            rationale="sandbox test",
            action="sandbox",
            instruction="Do something",
        )
        assert step.action == "sandbox"
        assert step.instruction == "Do something"
        assert step.tool_name is None

    def test_sandbox_step_missing_instruction(self):
        with pytest.raises(ValidationError, match="instruction is required"):
            Step(step_id="3", rationale="sandbox", action="sandbox", instruction=None)

    def test_parameters_defaults_to_empty_dict(self):
        step = Step(step_id="1", rationale="r", action="sandbox", instruction="i")
        assert step.parameters == {}

    def test_required_defaults_to_true(self):
        step = Step(step_id="1", rationale="r", action="sandbox", instruction="i")
        assert step.required is True

    def test_rationale_field_required(self):
        with pytest.raises(ValidationError):
            Step(step_id="1", action="sandbox", instruction="i")  # type: ignore

    def test_sanitization_in_parameters(self):
        import numpy as np
        step = Step(
            step_id="1",
            rationale="sanitize",
            action="tool",
            tool_name=ToolName.EVALUATE_METRICS,
            parameters={
                "expression": "x * 2",
                "values": {
                    "x": np.float64(3.14),   # will be sanitized to float 3.14
                }
            },
        )
        # After validation, the value should be a Python float
        assert step.parameters["values"]["x"] == 3.14
        assert isinstance(step.parameters["values"]["x"], float)
    
    def test_json_roundtrip(self):
        step = Step(
            step_id="1",
            rationale="roundtrip",
            action="tool",
            tool_name=ToolName.ANALYZE_DATA,
            parameters=VALID_PARAMS[ToolName.ANALYZE_DATA],
        )
        json_str = step.model_dump_json()
        restored = Step.model_validate_json(json_str)
        assert restored.step_id == step.step_id
        assert restored.parameters == step.parameters

    def test_unregistered_tool_name_raises_runtime_error(self, monkeypatch):
        monkeypatch.delitem(TOOL_PARAM_MODELS, ToolName.ANALYZE_DATA, raising=True)
        with pytest.raises(RuntimeError, match="not registered"):
            Step(step_id="1", rationale="x", action="tool",
                tool_name=ToolName.ANALYZE_DATA,
                parameters=VALID_PARAMS[ToolName.ANALYZE_DATA])

# ── StepRecord unit tests ──
class TestStepRecord:
    def test_success_record(self):
        rec = StepRecord(step_id="s1", tool_name=ToolName.ANALYZE_DATA, status="success")
        assert rec.status == "success"
        assert rec.summary is None

    def test_failed_record_with_error(self):
        rec = StepRecord(step_id="s2", tool_name=ToolName.CORRELATION, status="failed", error="Division by zero")
        assert rec.error == "Division by zero"
        assert rec.status == "failed"

    def test_skipped_record(self):
        rec = StepRecord(step_id="s3", status="skipped", summary="Skipped due to missing column")
        assert rec.status == "skipped"
        assert rec.tool_name is None

    def test_duration_ms(self):
        rec = StepRecord(step_id="s4", duration_ms=123)
        assert rec.duration_ms == 123

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            StepRecord(step_id="x", status="unknown")  # type: ignore


# ── Plan unit tests ──
class TestPlan:
    def test_empty_plan(self):
        plan = Plan(steps=[])
        assert plan.steps == []

    def test_non_empty_plan(self):
        steps = [
            Step(step_id="1", rationale="first", action="sandbox", instruction="do"),
            Step(step_id="2", rationale="second", action="tool", tool_name=ToolName.ANALYZE_DATA, parameters=VALID_PARAMS[ToolName.ANALYZE_DATA]),
        ]
        plan = Plan(steps=steps)
        assert len(plan.steps) == 2
        assert plan.steps[0].step_id == "1"

    def test_plan_json_roundtrip(self):
        plan = Plan(steps=[Step(step_id="1", rationale="test", action="sandbox", instruction="do")])
        json_str = plan.model_dump_json()
        restored = Plan.model_validate_json(json_str)
        assert restored.steps[0].rationale == "test"


# ── Hypothesis fuzzing for Step validator ──
@st.composite
def valid_step_strategy(draw):
    """Generate a Step that always passes its validator."""
    action = draw(st.sampled_from(["tool", "sandbox"]))
    step_id = draw(st.text(min_size=1, max_size=10))
    rationale = draw(st.text(min_size=1, max_size=50))
    required = draw(st.booleans())

    if action == "tool":
        tool_name = draw(st.sampled_from(list(ToolName)))
        parameters = VALID_PARAMS[tool_name]
        instruction = None
    else:
        tool_name = None
        parameters = {}
        instruction = draw(st.text(min_size=1, max_size=50))

    return Step(
        step_id=step_id,
        rationale=rationale,
        action=action,
        tool_name=tool_name,
        parameters=parameters,
        instruction=instruction,
        required=required,
    )


class TestStepFuzzing:
    @given(valid_step_strategy())
    def test_random_valid_step(self, step: Step):
        """Any step generated by our strategy should be valid and serializable."""
        json_str = step.model_dump_json()
        assert isinstance(json_str, str)
        restored = Step.model_validate_json(json_str)
        assert restored.step_id == step.step_id

    @given(
        st.text(),
        st.text(),
        st.sampled_from(["tool", "sandbox"]),
        st.none() | st.sampled_from(list(ToolName)),
        st.dictionaries(st.text(), st.text()),
        st.none() | st.text(),
        st.booleans(),
    )
    def test_invalid_step_combinations(
        self, step_id, rationale, action, tool_name, params, instruction, required
    ):
        """Random combinations should either construct successfully or raise ValidationError,
        never a raw Python exception."""
        try:
            step = Step(
                step_id=step_id,
                rationale=rationale,
                action=action,
                tool_name=tool_name,
                parameters=params,
                instruction=instruction,
                required=required,
            )
            # If construction succeeds, it must pass the validator and be serializable
            json_str = step.model_dump_json()
            assert isinstance(json_str, str)
        except ValidationError:
            pass  # expected for invalid combinations
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")


# ── Plan fuzzing ──
class TestPlanFuzzing:
    @given(st.lists(valid_step_strategy(), max_size=5))
    def test_random_plan(self, steps):
        plan = Plan(steps=steps)
        assert len(plan.steps) == len(steps)
        json_str = plan.model_dump_json()
        restored = Plan.model_validate_json(json_str)
        assert len(restored.steps) == len(steps)

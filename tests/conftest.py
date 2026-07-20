"""Root pytest configuration and shared fixture repository for Scrygent.

This conftest is the foundation for the entire test matrix. It provides:

- Deterministic AgentState factories wired to ephemeral CSV artifacts.
- Boundary-stress fixtures that inject NumPy/Pandas C-types to verify the
  ScrygentBaseModel JSON sanitization layer.
- A context manager that stubs `resilient_call` so the three-pass compiler
  (Planner → Executor → Reporter) can be exercised end-to-end without any
  network I/O or live LLM access.
"""

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scrygent.base_model import ScrygentBaseModel
from scrygent.models.state import AgentState

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _materialize_dummy_csv(path: Path) -> Path:
    """Write a minimal Titanic-style CSV to the specified path and return it.

    The schema includes integer, float (with NaN), and string columns so
    that Profiler, statistics, and wrangling contracts can be exercised
    without depending on the bundled `data/` directory.
    """
    pd.DataFrame({
        "passenger_id": pd.array([1, 2, 3, 4], dtype="int64"),
        "survived": pd.array([0, 1, 1, 0], dtype="int64"),
        "age": pd.array([22.0, 38.0, np.nan, 35.0], dtype="float64"),
        "fare": pd.array([7.25, 71.28, np.nan, 8.05], dtype="float64"),
        "embarked": pd.array(["S", "C", "S", "Q"], dtype=object),
    }).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_csv_path(tmp_path: Path) -> Path:
    """Provide a deterministic on-disk CSV for Profiler and tool I/O tests.

    The file is written under pytest's `tmp_path` so it is torn down
    automatically at session exit. The schema mirrors a minimal
    Titanic-style dataset so that statistics, filtering, and wrangling
    contracts can be exercised without depending on the bundled `data/`
    directory.
    """
    return _materialize_dummy_csv(tmp_path / "dummy.csv")


@pytest.fixture
def poisoned_csv_path(tmp_path: Path) -> Path:
    """Provide a semicolon-delimited UTF-16 CSV to exercise Profiler resilience.

    This mimics the `data/poisoned/titanic_semicolon.csv` and
    `titanic_utf16.csv` artifacts but is generated hermetically under
    `tmp_path`. Use it to assert that the Profiler detects encoding
    and delimiter anomalies and logs them to `error_log` rather than
    crashing.
    """
    path = tmp_path / "poisoned.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(path, index=False, sep=";", encoding="utf-16")
    return path


# ---------------------------------------------------------------------------
# AgentState fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_agent_state(dummy_csv_path: Path) -> AgentState:
    """Return a fully valid `AgentState` primed for a fresh execution cycle.

    The state is configured with:

    - `original_csv_path` == `current_csv_path` (no wrangling yet).
    - `execution_status` == `"pending"` (entry-node invariant).
    - `has_replanned` == `False` (the lazy-fetch guard is unspent).
    - `eval_mode` == `False` (Reporter will emit `AnalysisReport`).
    - `retry_count` == `0` and empty `step_outputs` / `error_log`.

    Use this as the baseline for unit tests on individual nodes and for
    integration tests that walk the graph from a cold start.
    """
    return AgentState(
        original_csv_path=dummy_csv_path,
        current_csv_path=dummy_csv_path,
        user_query="What is the average age of survivors?",
        eval_mode=False,
    )


@pytest.fixture
def eval_mode_agent_state(dummy_csv_path: Path) -> AgentState:
    """Return an `AgentState` configured for `eval_mode=True`.

    In eval mode the Reporter is contractually obligated to emit a
    `DirectAnswer` and must drop all narrative and plot artifacts.
    This fixture lets unit and integration tests assert that contract
    without flipping the flag inline.
    """
    return AgentState(
        original_csv_path=dummy_csv_path,
        current_csv_path=dummy_csv_path,
        user_query="How many passengers survived?",
        eval_mode=True,
    )


@pytest.fixture
def replan_guard_spent_state(valid_agent_state: AgentState) -> AgentState:
    """Return a state whose `has_replanned` guard has already been spent.

    The lazy-fetch `replan` back-edge is the only LangGraph back-edge
    in Scrygent and is gated by this flag. Tests asserting that the
    graph refuses to re-enter the Planner should seed from this fixture.
    """
    return valid_agent_state.model_copy(update={"has_replanned": True})


@pytest.fixture
def mid_execution_state(valid_agent_state: AgentState) -> AgentState:
    """Return a state mid-execution with one completed step.

    The state has `execution_status="running"`, `current_step_index=1`,
    and a single entry in `step_outputs` keyed by `"step_001"`. Use this
    for tests that need to assert on the Executor's continuation
    behavior or the Reporter's synthesis from partial results.
    """
    return valid_agent_state.model_copy(
        update={
            "execution_status": "running",
            "current_step_index": 1,
            "step_outputs": {"step_001": {"row_count": 4, "column_count": 5}},
        }
    )


# ---------------------------------------------------------------------------
# Hermetic JSON-boundary fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numpy_polluted_payload() -> dict[str, Any]:
    """Return a raw dict seeded with non-JSON-native NumPy and Pandas values.

    The payload deliberately exercises every branch of
    `_sanitize_scalar` in `scrygent.base_model`:

    - `np.int64`, `np.float64`, `np.bool_`: scalar C-types.
    - `pd.NaT` and `np.datetime64`: temporal sentinels.
    - `np.nan` / `np.inf`: floating-point sentinels (must become `None`).
    - `pd.Timestamp`: ISO-format conversion.
    - Nested `np.ndarray`: exercises `_recursive_sanitize`.
    - `np.int32`, `np.int8`, `np.float32`: alternate bit-widths.

    Use this to drive tests that construct `ScrygentBaseModel`
    subclasses and assert on the sanitized output.
    """
    return {
        "count": np.int64(42),
        "ratio": np.float64(3.14),
        "is_valid": np.bool_(True),
        "missing_time": pd.NaT,
        "event_time": np.datetime64("2024-01-01T00:00:00"),
        "float_nan": np.float64(np.nan),
        "float_inf": np.float64(np.inf),
        "timestamp": pd.Timestamp("2024-06-15T12:00:00"),
        "tags": np.array(["alpha", "beta"], dtype=object),
        "nested": {
            "inner_int": np.int32(7),
            "inner_list": [np.int8(1), np.float32(2.5), np.bool_(False)],
        },
    }


@pytest.fixture
def sanitized_counterpart() -> dict[str, Any]:
    """Return the canonical native-Python equivalent of `numpy_polluted_payload`.

    Pair this with `numpy_polluted_payload` in assertion-driven tests
    to verify the hermetic boundary produces stable, JSON-serializable
    output. Values are aligned field-by-field with the polluted fixture
    so a deep equality check is sufficient.
    """
    return {
        "count": 42,
        "ratio": 3.14,
        "is_valid": True,
        "missing_time": None,
        "event_time": "2024-01-01T00:00:00",
        "float_nan": None,
        "float_inf": None,
        "timestamp": "2024-06-15T12:00:00",
        "tags": ["alpha", "beta"],
        "nested": {
            "inner_int": 7,
            "inner_list": [1, 2.5, False],
        },
    }


class _BoundaryProbe(ScrygentBaseModel):
    """Minimal `ScrygentBaseModel` subclass used to probe the sanitization layer.

    The single `payload` field accepts an arbitrary dict so tests can
    round-trip polluted dictionaries through the
    `@model_validator(mode="wrap")` sanitizer without defining a schema
    per field. This isolates sanitization correctness from contract-level
    schema validation.
    """

    payload: dict[str, Any]


@pytest.fixture
def boundary_probe_model() -> type[_BoundaryProbe]:
    """Return a throwaway `ScrygentBaseModel` subclass for boundary tests.

    The probe model exposes a single `payload: dict[str, Any]` field so
    that tests can round-trip arbitrary polluted dictionaries through the
    `@model_validator(mode="wrap")` sanitizer. This isolates
    sanitization correctness from contract-level schema validation.
    """
    return _BoundaryProbe


@pytest.fixture
def polluted_pydantic_output(
    boundary_probe_model: type[_BoundaryProbe],
    numpy_polluted_payload: dict[str, Any],
) -> _BoundaryProbe:
    """Construct a `ScrygentBaseModel` instance from NumPy/Pandas-polluted data.

    This fixture is the primary entry point for tests asserting that the
    hermetic boundary scrubs C-types at instantiation time. The returned
    model's `model_dump()["payload"]` should be deep-equal to
    `sanitized_counterpart`.
    """
    return boundary_probe_model(payload=numpy_polluted_payload)


@pytest.fixture
def unsanitizable_payload() -> dict[str, Any]:
    """Return a payload that must raise `SanitizationError`.

    Includes a raw `pd.DataFrame` and `pd.Series`, which the recursive
    sanitizer explicitly rejects to enforce the stateless-tool contract.
    Use with `pytest.raises(SanitizationError)` to assert the negative
    boundary.
    """
    return {
        "leaked_frame": pd.DataFrame({"x": [1, 2]}),
        "leaked_series": pd.Series([1, 2, 3], name="y"),
    }


# ---------------------------------------------------------------------------
# resilient_call mocking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompiledResponse:
    """Type-safe carrier for a single deterministic `resilient_call` return.

    Attributes:
        payload: The JSON-native object to return (already validated
            against its schema by the test author).
        schema: Optional Pydantic schema the mock will assert the caller
            requested. Only enforced when `assert_schema=True` is passed
            to `mock_resilient_call`.
    """

    payload: Any
    schema: type | None = None


@dataclass(slots=True)
class _ResilientCallScript:
    """Internal FIFO queue and call recorder for `mock_resilient_call`."""

    remaining: list[Any] = field(default_factory=list)
    recorded_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        """Return the number of times the patched `resilient_call` was invoked."""
        return len(self.recorded_calls)


@contextmanager
def mock_resilient_call(
    responses: Sequence[Any],
    *,
    assert_schema: bool = False,
) -> Iterator[_ResilientCallScript]:
    """Stub `scrygent.resilience.resilient_call` with deterministic payloads.

    This is the single entry point for exercising the three-pass compiler
    (Planner → Executor → Reporter) without any network I/O. Each call
    to the patched `resilient_call` pops the next response from the
    queue:

    - If the response is a `BaseException` instance, it is raised,
      allowing tests to simulate transient API failures and self-healing
      retries within the Executor node.
    - If the response is a `CompiledResponse` with `schema` set and
      `assert_schema` is `True`, the mock validates that the caller
      requested that exact Pydantic schema before returning `payload`.
    - Otherwise, the raw response object is returned as-is.

    Args:
        responses: Ordered sequence of payloads, exceptions, or
            `CompiledResponse` objects the mock will yield.
        assert_schema: When `True`, enforce `CompiledResponse.schema`
            checks against the caller's requested schema.

    Yields:
        A `_ResilientCallScript` exposing `recorded_calls` and
        `call_count` for post-hoc assertions about how many times the
        compiler invoked `resilient_call` and with what arguments.

    Raises:
        AssertionError: If the compiler exhausts the response queue (an
            unexpected extra call) or, at context exit, if responses
            remain unconsumed (the compiler short-circuited a pass).
    """
    from unittest.mock import patch

    script = _ResilientCallScript(remaining=list(responses))

    def _fake_resilient_call(*args: Any, **kwargs: Any) -> Any:
        script.recorded_calls.append((args, dict(kwargs)))

        if not script.remaining:
            err = AssertionError("resilient_call was invoked more times than the test script provided responses for.")
            err.add_note(f"Recorded {script.call_count} call(s); only {len(responses)} response(s) were supplied.")
            raise err

        response = script.remaining.pop(0)

        if isinstance(response, BaseException):
            raise response

        if isinstance(response, CompiledResponse):
            if assert_schema and response.schema is not None:
                requested = kwargs.get("schema") or (args[0] if args else None)
                if requested is not response.schema:
                    err = AssertionError(
                        f"Schema mismatch: compiler requested {requested!r}, script expected {response.schema!r}."
                    )
                    err.add_note(
                        "Pass assert_schema=False to disable this check, "
                        "or align CompiledResponse.schema with the caller's request."
                    )
                    raise err
            return response.payload

        return response

    with (
        patch("scrygent.core.resilience.resilient_call", side_effect=_fake_resilient_call),
        patch("scrygent.agents.planner_node.resilient_call", side_effect=_fake_resilient_call),
        patch("scrygent.agents.executor_node.resilient_call", side_effect=_fake_resilient_call),
        patch("scrygent.agents.reporter_node.resilient_call", side_effect=_fake_resilient_call),
    ):
        try:
            yield script
        finally:
            if script.remaining:
                err = AssertionError(
                    f"resilient_call was invoked {script.call_count} time(s) "
                    f"but {len(responses)} response(s) were supplied; "
                    f"{len(script.remaining)} response(s) were never consumed."
                )
                err.add_note(
                    "An unconsumed response usually means the compiler "
                    "short-circuited a pass (e.g., an Executor self-healing "
                    "branch returned early, or the Planner emitted an empty plan)."
                )
                raise err


@pytest.fixture
def resilient_call_mock() -> Callable[..., AbstractContextManager[_ResilientCallScript]]:
    """Expose the `mock_resilient_call` context manager as a fixture.

    Returns the context manager itself (not an active context) so
    individual tests can parameterize the response script. This
    indirection keeps the response queue local to each test, preventing
    cross-test contamination while sharing the patching boilerplate.
    """
    return mock_resilient_call


# ---------------------------------------------------------------------------
# Environment & safety fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_request_pacer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the global LLM request pacer for test execution speed.

    The pacer (controlled by `SCRYGENT_PACE_REQUESTS`) enforces a
    minimum interval between real LLM calls. In tests, all LLM calls are
    mocked, so the pacer would only introduce artificial delays. This
    autouse fixture runs before every test.
    """
    monkeypatch.setenv("SCRYGENT_PACE_REQUESTS", "false")


@pytest.fixture
def mock_api_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject dummy API credentials for tests that construct LLM clients.

    Sets placeholder values for `GROQ_API_KEY` and `OPENROUTER_API_KEY`
    so that `_build_groq_llm` / `_build_openrouter_llm` do not raise
    `ValueError` during client construction. The credentials are never
    sent to a real endpoint because `resilient_call` is mocked.

    Tests that assert on missing credentials should not request this
    fixture.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real-0001")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real-0002")


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and Hypothesis profiles at collection time.

    Markers registered here are recognized by `--strict-markers`,
    preventing typos from silently passing. Hypothesis profiles are
    loaded from the `HYPOTHESIS_PROFILE` environment variable, defaulting
    to `"dev"` for fast iteration. Use `HYPOTHESIS_PROFILE=ci` in CI and
    `HYPOTHESIS_PROFILE=stress` for exhaustive fuzzing.
    """
    config.addinivalue_line(
        "markers",
        "stress: mark test as property-based fuzzing (deselect with '-m \"not stress\"').",
    )
    config.addinivalue_line(
        "markers",
        "hermetic: marks tests that must never touch the network or live LLMs.",
    )
    config.addinivalue_line(
        "markers",
        "boundary: marks tests targeting the ScrygentBaseModel JSON sanitization layer.",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests exercising multiple Scrygent nodes end-to-end.",
    )
    config.addinivalue_line(
        "markers",
        "graph: marks tests exercising LangGraph routing and conditional edges.",
    )

    try:
        from hypothesis import HealthCheck, settings

        settings.register_profile(
            "ci",
            max_examples=200,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
            print_blob=True,
        )
        settings.register_profile(
            "dev",
            max_examples=25,
            deadline=None,
            suppress_health_check=[
                HealthCheck.too_slow,
                HealthCheck.data_too_large,
            ],
        )
        settings.register_profile(
            "stress",
            max_examples=2000,
            deadline=None,
            suppress_health_check=list(HealthCheck),
        )
        settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
    except ImportError:
        pass


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-assign markers based on test-path conventions.

    Tests located under `tests/stress/` are automatically marked as `stress`
    so they are deselected by the default `-m "not stress"` filter. This avoids
    per-file decorator boilerplate while keeping stress tests segregated.
    """
    for item in items:
        if "stress" in item.path.parts:
            item.add_marker(pytest.mark.stress)

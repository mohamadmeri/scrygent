"""Destructive test suite for the network resilience layer.

This module aggressively tests the exponential-backoff wrapper and UI
cooldown state management. It ensures that transient rate limits are
retried deterministically, non-rate-limit errors crash immediately to
preserve the self-healing correction loop, and UI state flags toggle
exactly during the cooldown window.
"""

from typing import Any

import pytest

from scrygent.core.resilience import RetryEvent, ServiceExhaustedError, is_system_cooling_down, resilient_call


class RateLimitError(Exception):
    """Mock exception simulating a standard HTTP 429 error."""

    def __init__(self, message: str = "Rate limit exceeded", response: Any = None) -> None:
        super().__init__(message)
        self.response = response


class MockResponse:
    """Mock HTTP response containing headers."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


@pytest.fixture
def mock_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock `time.sleep` to prevent real test delays during backoff."""
    monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: None)


@pytest.fixture
def deterministic_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock `random.uniform` to remove jitter, ensuring exact delay assertions."""
    monkeypatch.setattr("scrygent.core.resilience.random.uniform", lambda a, b: 0.0)


class TestResilientCall:
    """Tests validating the retry logic, error classification, and state management."""

    def test_returns_value_on_first_success_without_sleeping(
        self,
        mock_sleep: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify a successful callable execution bypasses the retry loop entirely.

        Asserts `time.sleep` is never called and the exact return value is propagated.
        """
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: pytest.fail("Sleep should not be called"))

        def success_fn() -> str:
            return "success"

        result = resilient_call(success_fn)
        assert result == "success"

    def test_reraises_non_rate_limit_errors_immediately(
        self,
        mock_sleep: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a standard ValueError (e.g., IR validation failure).

        The wrapper must re-raise immediately without retrying to preserve
        the integrity of the Executor's self-healing correction loop.
        """
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: pytest.fail("Sleep should not be called"))

        def bad_fn() -> None:
            raise ValueError("IR validation failed")

        with pytest.raises(ValueError, match="IR validation failed"):
            resilient_call(bad_fn)

    def test_retries_on_rate_limit_and_succeeds(
        self,
        mock_sleep: None,
        deterministic_jitter: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a function that raises a 429 error once, then succeeds.

        Asserts the wrapper catches the error, calculates the exact exponential
        backoff (base_delay * 2^0), sleeps, and retries successfully.
        """
        sleep_calls: list[float] = []
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: sleep_calls.append(x))

        calls = iter([RateLimitError("429"), "success"])

        def flaky_fn() -> str:
            val = next(calls)
            if isinstance(val, Exception):
                raise val
            return val

        result = resilient_call(flaky_fn, base_delay=2.0)

        assert result == "success"
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 2.0  # 2.0 * 2^0 + 0.0 jitter

    def test_raises_service_exhausted_after_max_attempts(
        self,
        mock_sleep: None,
        deterministic_jitter: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a function that always raises a 429 error.

        Asserts the wrapper exhausts all attempts and raises a
        `ServiceExhaustedError` containing the exact service name and count.
        """
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: None)

        def always_fails_fn() -> None:
            raise RateLimitError("429")

        with pytest.raises(ServiceExhaustedError) as exc_info:
            resilient_call(always_fails_fn, service="Groq", max_attempts=3)

        assert "Groq did not recover after 3 attempt(s)" in str(exc_info.value)
        assert exc_info.value.service == "Groq"
        assert exc_info.value.attempts == 3

    def test_extracts_retry_after_from_groq_headers(
        self,
        mock_sleep: None,
        deterministic_jitter: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a 429 error containing Groq's `x-ratelimit-reset-requests` header.

        The wrapper must extract the server-suggested wait time (10.0s) and use
        it instead of the exponential backoff calculation.
        """
        sleep_calls: list[float] = []
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: sleep_calls.append(x))

        response = MockResponse(headers={"x-ratelimit-reset-requests": "10.0"})

        def header_fail_fn() -> None:
            raise RateLimitError("429", response=response)

        with pytest.raises(ServiceExhaustedError):
            resilient_call(header_fail_fn, max_attempts=2)

        assert sleep_calls[0] == 10.0  # Server time + 0.0 jitter

    def test_extracts_retry_after_from_error_string(
        self,
        mock_sleep: None,
        deterministic_jitter: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a 429 error containing a 'try again in 12.5s' message.

        The regex extractor must parse the float and use it for the sleep duration.
        """
        sleep_calls: list[float] = []
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: sleep_calls.append(x))

        def string_fail_fn() -> None:
            raise RateLimitError("Please try again in 12.5s")

        with pytest.raises(ServiceExhaustedError):
            resilient_call(string_fail_fn, max_attempts=2)

        assert sleep_calls[0] == 12.5


class TestCooldownState:
    """Tests validating the UI integration and cooldown flag management."""

    def test_cooldown_flag_toggles_during_retry_and_resets(
        self,
        mock_sleep: None,
        deterministic_jitter: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a function that raises a 429 error, then succeeds.

        Asserts `is_system_cooling_down()` is True during the `on_retry` handler
        execution, and returns to False after the wrapper recovers.
        """
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: None)
        cooldown_states_during_retry: list[bool] = []

        def on_retry(event: RetryEvent) -> None:
            cooldown_states_during_retry.append(is_system_cooling_down())

        calls = iter([RateLimitError("429"), "success"])

        def flaky_fn() -> str:
            val = next(calls)
            if isinstance(val, Exception):
                raise val
            return val

        assert is_system_cooling_down() is False

        result = resilient_call(flaky_fn, max_attempts=2, on_retry=on_retry)

        assert result == "success"
        assert is_system_cooling_down() is False
        assert len(cooldown_states_during_retry) == 1
        assert cooldown_states_during_retry[0] is True

    def test_custom_on_retry_handler_receives_exact_event(
        self,
        mock_sleep: None,
        deterministic_jitter: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a function that raises a 429 error, then succeeds.

        Asserts the `on_retry` callback receives a `RetryEvent` dataclass
        containing the exact service, attempt, and wait_seconds attributes.
        """
        monkeypatch.setattr("scrygent.core.resilience.time.sleep", lambda x: None)
        received_events: list[RetryEvent] = []

        def on_retry(event: RetryEvent) -> None:
            received_events.append(event)

        calls = iter([RateLimitError("429"), "success"])

        def flaky_fn() -> str:
            val = next(calls)
            if isinstance(val, Exception):
                raise val
            return val

        resilient_call(flaky_fn, service="OpenRouter", max_attempts=2, base_delay=4.0, on_retry=on_retry)

        assert len(received_events) == 1
        event = received_events[0]
        assert event.service == "OpenRouter"
        assert event.attempt == 1
        assert event.max_attempts == 2
        assert event.wait_seconds == 4.0  # 4.0 * 2^0 + 0.0 jitter
        assert isinstance(event.error, RateLimitError)

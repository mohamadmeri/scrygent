"""Network resilience and rate-limit management for outbound LLM calls.

This module provides a deterministic exponential-backoff wrapper for
HTTP 429 (Too Many Requests) errors. It isolates network instability
from the core execution graph, ensuring that transient rate limits
do not trigger false-positive schema validation failures.

The module utilizes contextvars to inject UI-level cooldown callbacks
without importing Streamlit, maintaining strict architectural isolation.
"""

from __future__ import annotations

import contextvars
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryEvent:
    """Snapshot of a retry attempt, passed to UI callbacks for cooldown rendering."""

    service: str
    attempt: int
    max_attempts: int
    wait_seconds: float
    error: Exception


class ServiceExhaustedError(RuntimeError):
    """Raised when all retry attempts are exhausted without recovery."""

    def __init__(self, service: str, attempts: int, last_error: Exception):  # noqa: D107
        self.service = service
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"{service} did not recover after {attempts} attempt(s): {last_error}")


_retry_handler: contextvars.ContextVar[Callable[[RetryEvent], None] | None] = contextvars.ContextVar(
    "scrygent_retry_handler", default=None
)


def set_retry_handler(handler: Callable[[RetryEvent], None] | None) -> None:
    """Registers a callback invoked on every retry attempt. Typically called by the UI layer."""
    _retry_handler.set(handler)


def get_retry_handler() -> Callable[[RetryEvent], None] | None:
    """Retrieves the currently registered retry handler, if any."""
    return _retry_handler.get()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Determines if an exception represents an HTTP 429 rate limit violation."""
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429:
        return True
    return exc.__class__.__name__ in {"RateLimitError", "TooManyRequestsError"}


def _extract_retry_after(exc: Exception) -> float | None:
    """Extracts the server-suggested wait time from a 429 response.

    Checks standard headers (retry-after, x-ratelimit-reset-after) and
    falls back to parsing embedded error messages for Groq compatibility.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
            if key in headers:
                try:
                    return float(headers[key])
                except (TypeError, ValueError):
                    pass

    match = re.search(r"try again in ([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def resilient_call(
    fn: Callable[[], T],
    *,
    service: str = "Groq",
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    on_retry: Callable[[RetryEvent], None] | None = None,
) -> T:
    """Executes a callable with exponential-backoff retry on HTTP 429 rate limits.

    Non-429 exceptions are re-raised immediately to preserve the integrity
    of the Self-Healing Correction Loop. If a retry handler is registered,
    it assumes control of the wait duration to enable live UI countdowns;
    otherwise, the wrapper performs a synchronous sleep.

    Args:
        fn: The zero-argument callable to execute.
        service: Identifier for the target service (used in logging and errors).
        max_attempts: Maximum number of execution attempts before failure.
        base_delay: Initial delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
        on_retry: Optional explicit callback for retry events.

    Returns:
        The result of the successful callable execution.

    Raises:
        ServiceExhaustedError: If all attempts fail with rate limit errors.
        Exception: Any non-rate-limit exception raised by the callable.
    """
    handler = on_retry or get_retry_handler()
    attempt = 0
    last_error: Exception | None = None

    while attempt < max_attempts:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise

            last_error = exc
            if attempt >= max_attempts:
                break

            wait = _extract_retry_after(exc)
            if wait is None:
                wait = min(base_delay * (2 ** (attempt - 1)), max_delay)

            # Add jitter to prevent thundering herd on shared API quotas
            wait += random.uniform(0, 0.5)

            logger.warning(
                "%s rate limited (attempt %d/%d). Cooling down for %.1fs.",
                service,
                attempt,
                max_attempts,
                wait,
            )

            if handler:
                handler(
                    RetryEvent(
                        service=service,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        wait_seconds=wait,
                        error=exc,
                    )
                )
            else:
                time.sleep(wait)

    # last_error is guaranteed to be set if we exit the loop without returning
    raise ServiceExhaustedError(service, max_attempts, last_error)  # type: ignore[arg-type]

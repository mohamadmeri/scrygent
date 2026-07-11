"""
Resilience layer for Scrygent — Smart Resilient Wrapper for outbound LLM calls.

Per the Dependency Golden Rule in ARCHITECTURE.md, this module lives below
`agents/` and has ZERO Streamlit / UI imports. It has no idea a UI exists.
Instead, it exposes retry progress through an `on_retry` callback and an
optional `contextvars.ContextVar` "sink" that the UI layer (app.py) can
register before invoking the graph. This keeps the retry logic reusable
and unit-testable outside of Streamlit entirely.
"""

from __future__ import annotations

import contextvars
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ==============================================================================
# EVENTS & ERRORS
# ==============================================================================

@dataclass
class RetryEvent:
    """Snapshot handed to the on_retry callback so a UI can render a cooldown state."""
    service: str
    attempt: int
    max_attempts: int
    wait_seconds: float
    error: Exception


class ServiceExhaustedError(RuntimeError):
    """Raised when every retry attempt has been burned (e.g. persistent 429s)."""

    def __init__(self, service: str, attempts: int, last_error: Exception):
        self.service = service
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"{service} did not recover after {attempts} attempt(s): {last_error}")


# ==============================================================================
# UI CALLBACK REGISTRY (contextvar — no streamlit import required here)
# ==============================================================================

_retry_handler: contextvars.ContextVar[Optional[Callable[[RetryEvent], None]]] = (
    contextvars.ContextVar("scrygent_retry_handler", default=None)
)


def set_retry_handler(handler: Optional[Callable[[RetryEvent], None]]) -> None:
    """Registers a callback invoked on every retry attempt. Call from app.py before graph.invoke()."""
    _retry_handler.set(handler)


def get_retry_handler() -> Optional[Callable[[RetryEvent], None]]:
    return _retry_handler.get()


# ==============================================================================
# ERROR INTROSPECTION
# ==============================================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status_code == 429:
        return True
    return exc.__class__.__name__ in {"RateLimitError", "TooManyRequestsError"}


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """
    Best-effort extraction of a server-suggested wait time from a 429.

    Groq (OpenAI-compatible) surfaces this in one of two places depending on
    client version:
      1. exc.response.headers["retry-after"]
      2. An embedded "Please try again in 12.3s" style message
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


# ==============================================================================
# THE WRAPPER
# ==============================================================================

def resilient_call(
    fn: Callable[[], T],
    *,
    service: str = "Groq",
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable[[RetryEvent], None]] = None,
) -> T:
    """
    Executes `fn` with exponential-backoff retry on HTTP 429 rate limits.

    - Non-429 exceptions are re-raised immediately. This wrapper exists to
      smooth over rate limiting only — genuine failures (bad params, schema
      violations) are still the job of the Self-Healing Correction Loop.
    - If `on_retry` is not passed explicitly, falls back to whatever handler
      was registered via `set_retry_handler` (typically wired to the UI).
    - IMPORTANT: when a handler is registered, it OWNS the wait. This lets it
      render a live countdown (progress bar, "12s remaining"...) instead of
      the wrapper silently sleeping. If no handler is registered, the wrapper
      does a plain `time.sleep(wait)` itself so headless/CI use still works.
    - Raises ServiceExhaustedError after `max_attempts`, never hangs forever.
    """
    handler = on_retry or get_retry_handler()
    attempt = 0
    last_error: Optional[Exception] = None

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
            wait += random.uniform(0, 0.5)  # jitter, avoids thundering herd on shared quotas

            logger.warning(
                "%s rate limited (attempt %d/%d). Cooling down for %.1fs.",
                service, attempt, max_attempts, wait,
            )

            if handler:
                handler(RetryEvent(
                    service=service,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    wait_seconds=wait,
                    error=exc,
                ))
            else:
                time.sleep(wait)

    raise ServiceExhaustedError(service, max_attempts, last_error) # type: ignore

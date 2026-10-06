"""Retries for every request a wire sends, at the HTTP layer: backoff with full jitter.

Only what cannot have been billed is retried: a refusal status, or a connection that never reached
the server. Timeouts and dropped responses are not. The SDKs' own retries are off.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import httpx2

log = logging.getLogger("aigent.retry")

# 529 is Anthropic's "overloaded".
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class Retry:
    """One retry about to happen: which attempt failed, why, and how long until the next."""

    attempt: int
    reason: str
    delay: float


def wait_for(
    retry_after: str | None, attempt: int, base: float, cap: float, rng: random.Random
) -> float | None:
    """Seconds before retry `attempt` (from 1): the server's `retry-after` if it gave one, else a
    full-jitter draw under `base * 2**(attempt - 1)` and `cap`. None when the server asks for more
    than `cap`, which is not worth waiting for."""
    asked = _seconds(retry_after)
    if asked is not None:
        return asked if asked <= cap else None
    return rng.uniform(0.0, min(cap, base * 2 ** (attempt - 1)))


def _seconds(retry_after: str | None) -> float | None:
    """`retry-after` as seconds, whether it was sent as a number or an HTTP date."""
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


class RetryingTransport(httpx2.BaseTransport):
    """`inner` with up to `retries` more attempts for each request it could safely send again."""

    def __init__(
        self,
        inner: httpx2.BaseTransport,
        *,
        retries: int,
        base_seconds: float,
        max_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        on_retry: Callable[[Retry], None] | None = None,
    ) -> None:
        self._inner = inner
        self._retries = retries
        self._base = base_seconds
        self._cap = max_seconds
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._on_retry = on_retry

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        request.read()  # the body must be in memory to be sent twice
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._inner.handle_request(request)
            except (httpx2.ConnectError, httpx2.ConnectTimeout) as refused:
                delay = self._delay(attempt, None)
                if delay is None:
                    raise
                self._wait(Retry(attempt, type(refused).__name__, delay))
                continue
            if not _retryable(response):
                return response
            delay = self._delay(attempt, response.headers.get("retry-after"))
            if delay is None:
                return response
            response.close()
            self._wait(Retry(attempt, f"HTTP {response.status_code}", delay))

    def _delay(self, attempt: int, retry_after: str | None) -> float | None:
        """How long to wait before trying again, or None once there is no retry left to try."""
        if attempt > self._retries:
            return None
        return wait_for(retry_after, attempt, self._base, self._cap, self._rng)

    def _wait(self, retry: Retry) -> None:
        log.warning(
            "attempt %d failed (%s); retrying in %.1fs", retry.attempt, retry.reason, retry.delay
        )
        if self._on_retry is not None:
            self._on_retry(retry)
        self._sleep(retry.delay)

    def close(self) -> None:
        self._inner.close()


def _retryable(response: httpx2.Response) -> bool:
    """A refusal worth repeating; the server's own `x-should-retry` wins when it sends one."""
    told = response.headers.get("x-should-retry")
    if told is not None:
        return told == "true"
    return response.status_code in RETRY_STATUSES


def http_client(
    timeout: float,
    keepalive_seconds: float,
    retries: int,
    base_seconds: float,
    max_seconds: float,
) -> httpx2.Client:
    """The HTTP client a wire's SDK client sends through: this module's retries, its connection
    reuse and its timeout."""
    inner = httpx2.HTTPTransport(limits=httpx2.Limits(keepalive_expiry=keepalive_seconds))
    transport = RetryingTransport(
        inner, retries=retries, base_seconds=base_seconds, max_seconds=max_seconds
    )
    return httpx2.Client(transport=transport, timeout=timeout)

"""Retries: what is sent again, how long it waits, and that the SDKs leave the retrying to us.

Every test answers from `httpx2.MockTransport` and sleeps into a list, so none waits or touches the
network.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import anthropic
import httpx2
import pytest

from aigent.adapters.retry import Retry, RetryingTransport, wait_for

URL = "https://api.example/v1/messages"
Answer = int | tuple[int, dict[str, str]] | Exception


class Ceiling(random.Random):
    """A generator whose every uniform draw is the top of its range, so the ceiling is testable."""

    def uniform(self, a: float, b: float) -> float:
        return b


@pytest.mark.parametrize(
    ("retry_after", "attempt", "expected"),
    [
        pytest.param(None, 1, 1.0, id="the first wait is at most the base"),
        pytest.param(None, 3, 4.0, id="it doubles each attempt"),
        pytest.param(None, 10, 30.0, id="and never passes the cap"),
        pytest.param("7", 1, 7.0, id="the server's retry-after wins"),
        pytest.param("0.5", 4, 0.5, id="even when shorter than the backoff"),
        pytest.param("120", 1, None, id="longer than the cap is not worth waiting for"),
        pytest.param("soon", 2, 2.0, id="an unreadable retry-after falls back to backoff"),
    ],
)
def test_the_wait_before_a_retry(
    retry_after: str | None, attempt: int, expected: float | None
) -> None:
    assert wait_for(retry_after, attempt, 1.0, 30.0, Ceiling()) == expected


def test_a_retry_after_sent_as_a_date_is_read_as_seconds_from_now() -> None:
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=10), usegmt=True)
    waited = wait_for(later, 1, 1.0, 30.0, Ceiling())
    assert waited is not None and 8.0 <= waited <= 10.0, waited


def _replaying(answers: Sequence[Answer], seen: list[httpx2.Request]) -> httpx2.MockTransport:
    """A server answering each request with the next of `answers`: a status, a status with
    headers, or an exception raised as the connection's own."""
    queue = list(answers)

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, headers = answer if isinstance(answer, tuple) else (answer, {})
        return httpx2.Response(status, headers=headers, json={"ok": status == 200})

    return httpx2.MockTransport(handle)


def _client(
    answers: Sequence[Answer], seen: list[httpx2.Request], slept: list[float], retries: int = 3
) -> httpx2.Client:
    transport = RetryingTransport(
        _replaying(answers, seen),
        retries=retries,
        base_seconds=1.0,
        max_seconds=30.0,
        sleep=slept.append,
        rng=Ceiling(),
    )
    return httpx2.Client(transport=transport)


def _refused(name: str) -> Callable[[], Exception]:
    return lambda: getattr(httpx2, name)("no route", request=httpx2.Request("POST", URL))


@pytest.mark.parametrize(
    ("answers", "status", "attempts"),
    [
        pytest.param([200], 200, 1, id="an answer is not retried"),
        pytest.param([529, 200], 200, 2, id="overloaded, then answered"),
        pytest.param([429, 503, 502, 200], 200, 4, id="three refusals fit in three retries"),
        pytest.param([429, 429, 429, 429], 429, 4, id="the fourth refusal is handed back"),
        pytest.param([400, 200], 400, 1, id="a bad request is never retried"),
        pytest.param([(500, {"x-should-retry": "false"}), 200], 500, 1, id="the server says no"),
        pytest.param([(409, {"x-should-retry": "true"}), 200], 200, 2, id="the server says yes"),
        pytest.param([(429, {"retry-after": "120"}), 200], 429, 1, id="too long to wait"),
    ],
)
def test_which_answers_are_tried_again(answers: list[Answer], status: int, attempts: int) -> None:
    seen: list[httpx2.Request] = []
    slept: list[float] = []

    response = _client(answers, seen, slept).post(URL, json={"q": 1})

    assert response.status_code == status
    assert len(seen) == attempts, [r.url for r in seen]
    assert len(slept) == attempts - 1, slept


@pytest.mark.parametrize(
    ("failure", "retried"),
    [
        pytest.param("ConnectError", True, id="a refused connection never reached the server"),
        pytest.param("ConnectTimeout", True, id="nor did one that never connected"),
        pytest.param("ReadTimeout", False, id="a read timeout may already have been billed"),
        pytest.param("RemoteProtocolError", False, id="so may a dropped response"),
    ],
)
def test_only_a_failure_before_the_server_saw_it_is_retried(failure: str, retried: bool) -> None:
    seen: list[httpx2.Request] = []
    slept: list[float] = []
    client = _client([_refused(failure)(), 200], seen, slept)

    if retried:
        assert client.post(URL, json={"q": 1}).status_code == 200
    else:
        with pytest.raises(getattr(httpx2, failure)):
            client.post(URL, json={"q": 1})
    assert len(seen) == (2 if retried else 1)


def test_a_retry_sends_the_same_body_and_is_reported_before_it_waits() -> None:
    seen: list[httpx2.Request] = []
    reported: list[Retry] = []
    transport = RetryingTransport(
        _replaying([(529, {"retry-after": "2"}), 200], seen),
        retries=3,
        base_seconds=1.0,
        max_seconds=30.0,
        sleep=lambda _: None,
        on_retry=reported.append,
    )

    httpx2.Client(transport=transport).post(URL, json={"task": "pt-001"})

    assert seen[0].content == seen[1].content == b'{"task":"pt-001"}'
    assert reported == [Retry(attempt=1, reason="HTTP 529", delay=2.0)]


def test_the_sdk_leaves_the_retrying_to_the_transport() -> None:
    """Through the real Anthropic SDK: one overloaded answer, then a message, and the call
    succeeds on our retry rather than the SDK's."""
    seen: list[httpx2.Request] = []
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    answers = iter(
        [httpx2.Response(529, json={"type": "error"}), httpx2.Response(200, json=message)]
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return next(answers)

    transport = RetryingTransport(
        httpx2.MockTransport(handle),
        retries=3,
        base_seconds=1.0,
        max_seconds=30.0,
        sleep=lambda _: None,
    )
    sdk = anthropic.Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=transport)
    )

    reply = sdk.messages.create(
        model="claude-sonnet-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    )

    assert reply.content[0].type == "text" and len(seen) == 2

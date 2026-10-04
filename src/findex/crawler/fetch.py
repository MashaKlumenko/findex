"""One polite GET: a hard timeout, and a short retry budget for transient failures."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from findex.crawler.stats import CrawlStats

# Honest identity. The contact URL is this project's repository.
USER_AGENT = (
    "findex-crawler/0.6 (+https://github.com/MashaKlumenko/findex; educational crawler)"
)

# First retry waits about base seconds; each later retry doubles, then jitter.
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_CAP = 8.0


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One try against ``url``. ``error`` is set when this try did not succeed."""

    status: int | None
    nbytes: int
    elapsed: float
    error: str | None
    retry_after: float | None
    at: str = field(default_factory=_stamp)


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Final outcome after at most ``max_attempts`` tries."""

    url: str
    status: int | None
    body: bytes
    elapsed: float
    error: str | None
    attempts: tuple[Attempt, ...]
    content_type: str | None = None
    final_url: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.status is not None
            and 200 <= self.status < 300
        )


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds to wait. ``Retry-After`` is either delta-seconds or an HTTP date."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    clock = time.time() if now is None else now
    return max(0.0, when.timestamp() - clock)


def backoff_seconds(
    attempt: int,
    retry_after: float | None,
    *,
    base: float = DEFAULT_BACKOFF_BASE,
    cap: float = DEFAULT_BACKOFF_CAP,
    jitter: float = 0.0,
) -> float:
    """Exponential backoff plus jitter, and never shorter than ``Retry-After``.

    ``attempt`` is 0 for the first retry. The lab formula is
    ``min(cap, base * 2**attempt) + jitter``.
    """
    delay = min(cap, base * (2**attempt)) + jitter
    if retry_after is None:
        return delay
    return max(delay, retry_after)


def _default_jitter() -> float:
    return random.uniform(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class _Outcome:
    attempt: Attempt
    status: int | None
    body: bytes
    error: str | None
    content_type: str | None
    final_url: str
    done: bool


def _failed(
    *,
    elapsed: float,
    error: str,
    url: str,
    done: bool,
    retry_after: float | None = None,
) -> _Outcome:
    return _Outcome(
        attempt=Attempt(None, 0, elapsed, error, retry_after),
        status=None,
        body=b"",
        error=error,
        content_type=None,
        final_url=url,
        done=done,
    )


async def _one_attempt(
    client: httpx.AsyncClient, url: str, *, timeout: float
) -> _Outcome:
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            response = await client.get(url)
    except TimeoutError:
        return _failed(
            elapsed=time.perf_counter() - started, error="timeout", url=url, done=False
        )
    except httpx.TimeoutException:
        return _failed(
            elapsed=time.perf_counter() - started, error="timeout", url=url, done=False
        )
    except httpx.TooManyRedirects:
        return _failed(
            elapsed=time.perf_counter() - started,
            error="redirect_loop",
            url=url,
            done=True,
        )
    except httpx.HTTPError as exc:
        # Resets and DNS failures are not retried. Only 429, 5xx, and timeouts are.
        return _failed(
            elapsed=time.perf_counter() - started,
            error=type(exc).__name__,
            url=url,
            done=True,
        )

    elapsed = time.perf_counter() - started
    status = response.status_code
    body = response.content
    retry_after = parse_retry_after(response.headers.get("retry-after"))
    content_type = response.headers.get("content-type")
    final_url = str(response.url)
    if status == 429 or status >= 500:
        error = f"http_{status}"
        return _Outcome(
            attempt=Attempt(status, len(body), elapsed, error, retry_after),
            status=status,
            body=body,
            error=error,
            content_type=content_type,
            final_url=final_url,
            done=False,
        )
    error = None if 200 <= status < 300 else f"http_{status}"
    return _Outcome(
        attempt=Attempt(status, len(body), elapsed, error, retry_after),
        status=status,
        body=body,
        error=error,
        content_type=content_type,
        final_url=final_url,
        done=True,
    )


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = 3,
    base: float = DEFAULT_BACKOFF_BASE,
    cap: float = DEFAULT_BACKOFF_CAP,
    jitter: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    stats: CrawlStats | None = None,
) -> FetchResult:
    """GET ``url``. Retry 429, 5xx, and timeouts at most ``max_attempts`` times total.

    Other 4xx responses are returned immediately. A redirect loop is not retried.
    ``CancelledError`` is never swallowed: that is how ``TaskGroup`` stops workers.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    roll = _default_jitter if jitter is None else jitter
    attempts: list[Attempt] = []

    for index in range(max_attempts):
        if stats is not None:
            stats.note_request_started()
        try:
            outcome = await _one_attempt(client, url, timeout=timeout)
        finally:
            if stats is not None:
                stats.note_request_finished()

        attempts.append(outcome.attempt)
        if outcome.done or index + 1 >= max_attempts:
            return FetchResult(
                url=url,
                status=outcome.status,
                body=outcome.body,
                elapsed=outcome.attempt.elapsed,
                error=outcome.error,
                attempts=tuple(attempts),
                content_type=outcome.content_type,
                final_url=outcome.final_url,
            )
        await sleep(
            backoff_seconds(
                index,
                outcome.attempt.retry_after,
                base=base,
                cap=cap,
                jitter=roll(),
            )
        )

    # ``max_attempts >= 1`` and the loop returns on the last try. This is unreachable.
    raise RuntimeError("fetch ended without a result")

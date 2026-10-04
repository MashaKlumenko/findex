"""Frontier, workers, and a stream of pages.

``crawl`` is an async generator: the caller writes each page out and drops it.
Nothing in here tokenizes or builds postings. That work has no ``await`` in it,
so doing it on this stack would freeze every other request.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.log import CrawlLog
from findex.crawler.parse import is_html, parse_html
from findex.crawler.robots import Jitter, RobotsCache, Sleep
from findex.crawler.stats import CrawlStats
from findex.crawler.urls import canonicalize, hostname_of

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    """One HTML document, ready for ``iter_documents``."""

    url: str
    title: str
    text: str
    fetched_at: str


@dataclass
class _Ctrl:
    stop: bool = False
    scheduled: int = 0
    finished: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    shutdown: asyncio.Event = field(default_factory=asyncio.Event)


async def crawl(
    seeds: list[str] | tuple[str, ...],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: list[str] | None = None,
    min_delay: float = 0.2,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    log: CrawlLog | None = None,
    sleep: Sleep | None = None,
    jitter: Jitter | None = None,
    backoff_base: float = 0.5,
    backoff_cap: float = 8.0,
) -> AsyncIterator[Page]:
    """Yield up to ``max_pages`` pages, then cancel the workers and close the client.

    The frontier is an ``asyncio.Queue``. ``concurrency`` workers run inside one
    ``TaskGroup``. A global semaphore caps in-flight fetches; each host has its
    own semaphore and a minimum gap between requests. ``seen`` stores canonical
    URLs so a fragment or a reshuffled query is not downloaded twice.

    When ``allowed_domains`` is omitted, only the seed hostnames are followed.
    """
    if max_pages <= 0 or not seeds:
        return
    if concurrency < 1 or per_host < 1:
        raise ValueError("concurrency and per_host must be >= 1")

    stats = stats or CrawlStats()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            max_redirects=10,
            timeout=None,
        )
    assert client is not None

    pause = asyncio.sleep if sleep is None else sleep
    frontier: asyncio.Queue[str] = asyncio.Queue()
    stats.frontier = frontier
    ready: asyncio.Queue[Page] = asyncio.Queue()
    ctrl = _Ctrl()
    seen: set[str] = set()
    allow: set[str] = set()
    if allowed_domains:
        allow = {host.lower() for host in allowed_domains}
    if not allow:
        allow = {hostname_of(canonicalize(seed)) for seed in seeds}
        allow.discard("")

    def enqueue(url: str) -> None:
        if ctrl.stop or ctrl.scheduled >= max_pages:
            return
        canon = canonicalize(url)
        parts = urlsplit(canon)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return
        if parts.hostname.lower() not in allow:
            return
        if canon in seen:
            return
        seen.add(canon)
        ctrl.scheduled += 1
        frontier.put_nowait(canon)

    for seed in seeds:
        enqueue(seed)
    if ctrl.scheduled == 0:
        if owns_client:
            await client.aclose()
        return

    robots = RobotsCache(
        client,
        user_agent,
        timeout=timeout,
        stats=stats,
        log=log,
        sleep=sleep,
        jitter=jitter,
        base=backoff_base,
        cap=backoff_cap,
    )
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    last_start: dict[str, float] = {}

    async def pace(host: str) -> None:
        if min_delay <= 0:
            return
        lock = host_locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            host_locks[host] = lock
        async with lock:
            loop = asyncio.get_running_loop()
            wait_for = last_start.get(host, 0.0) + min_delay - loop.time()
            if wait_for > 0:
                await pause(wait_for)
            last_start[host] = loop.time()

    def host_sem(host: str) -> asyncio.Semaphore:
        sem = host_sems.get(host)
        if sem is None:
            sem = asyncio.Semaphore(per_host)
            host_sems[host] = sem
        return sem

    async def handle(url: str) -> None:
        host = hostname_of(url)
        async with global_sem, host_sem(host):
            # Pace only in front of a real request. A cached robots.txt is free.
            if not robots.loaded(url):
                await pace(host)
            permitted = await robots.allowed(url)
            if robots.failed(url):
                stats.errors += 1
                if log is not None:
                    log.note(url, "robots", "unavailable")
                return
            if not permitted:
                if log is not None:
                    log.note(url, "robots", "disallowed")
                return
            await pace(host)
            result = await _fetch_page(
                client,
                url,
                timeout=timeout,
                stats=stats,
                sleep=pause,
                jitter=jitter,
                base=backoff_base,
                cap=backoff_cap,
            )
        _log_attempts(log, url, result)
        if not result.ok:
            stats.errors += 1
            return
        final = canonicalize(result.final_url or url)
        if hostname_of(final) not in allow:
            return
        seen.add(final)
        if not is_html(result.content_type, result.body):
            return
        # Parsing is CPU. Off the loop so other fetches keep progressing.
        title, text, links = await asyncio.to_thread(
            _decode_and_parse, final, result.body
        )
        if not text.strip():
            return
        for link in links:
            enqueue(link)
        ready.put_nowait(
            Page(
                url=final,
                title=title,
                text=text,
                fetched_at=datetime.now(UTC).isoformat(),
            )
        )

    async def worker() -> None:
        while not ctrl.stop:
            url = await frontier.get()
            try:
                if ctrl.stop:
                    return
                await handle(url)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker failed on %s", url)
                stats.errors += 1
            finally:
                ctrl.finished += 1
                if ctrl.finished >= ctrl.scheduled:
                    ctrl.done.set()

    async def supervise() -> None:
        # The group lives in its own task so closing this generator (a break
        # from ``async for`` raises GeneratorExit) does not become an error
        # inside the group. Shutdown cancels every worker, then the group waits
        # until those cancellations finish.
        async with asyncio.TaskGroup() as tg:
            workers = [
                tg.create_task(worker(), name=f"crawl-worker-{index}")
                for index in range(concurrency)
            ]
            await ctrl.shutdown.wait()
            for worker_task in workers:
                worker_task.cancel()

    supervisor = asyncio.create_task(supervise(), name="crawl-supervisor")
    try:
        yielded = 0
        while yielded < max_pages:
            page = await _next_page(ready, ctrl)
            if page is None:
                break
            yielded += 1
            stats.pages = yielded
            yield page
    finally:
        ctrl.stop = True
        ctrl.done.set()
        ctrl.shutdown.set()
        try:
            await supervisor
        finally:
            stats.frontier = None
            if owns_client:
                await client.aclose()


async def _next_page(ready: asyncio.Queue[Page], ctrl: _Ctrl) -> Page | None:
    """The next parsed page, or ``None`` when the frontier is exhausted."""
    if not ready.empty():
        return ready.get_nowait()
    if ctrl.done.is_set():
        return None
    getter: asyncio.Task[Page] = asyncio.create_task(ready.get())
    waiter = asyncio.create_task(ctrl.done.wait())
    try:
        await asyncio.wait({getter, waiter}, return_when=asyncio.FIRST_COMPLETED)
        page = _page_from(getter)
        if page is None and not getter.done():
            getter.cancel()
        if page is not None:
            return page
        return _drain(ready)
    finally:
        if not waiter.done():
            waiter.cancel()
        if not getter.done():
            getter.cancel()
        await asyncio.gather(getter, waiter, return_exceptions=True)


def _page_from(task: asyncio.Task[Page]) -> Page | None:
    if not task.done() or task.cancelled():
        return None
    if task.exception() is not None:
        return None
    return task.result()


def _drain(ready: asyncio.Queue[Page]) -> Page | None:
    if ready.empty():
        return None
    return ready.get_nowait()


def _decode_and_parse(url: str, body: bytes) -> tuple[str, str, tuple[str, ...]]:
    html = body.decode("utf-8", errors="replace")
    return parse_html(url, html)


def _log_attempts(log: CrawlLog | None, url: str, result: FetchResult) -> None:
    if log is None:
        return
    for number, record in enumerate(result.attempts, start=1):
        log.attempt(url, number, record)


async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float,
    stats: CrawlStats,
    sleep: Callable[[float], Awaitable[None]],
    jitter: Jitter | None,
    base: float,
    cap: float,
) -> FetchResult:
    return await fetch(
        client,
        url,
        timeout=timeout,
        stats=stats,
        sleep=sleep,
        jitter=jitter,
        base=base,
        cap=cap,
    )

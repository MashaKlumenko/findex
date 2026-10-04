"""Fetch and cache ``robots.txt`` once per origin."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT, fetch
from findex.crawler.log import CrawlLog
from findex.crawler.stats import CrawlStats
from findex.crawler.urls import origin_of

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[], float]


class RobotsCache:
    """One ``robots.txt`` per origin. A 404 means the host allows every URL.

    A timeout or a 5xx after retries fails closed: we do not crawl a host
    whose rules we could not read.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str = USER_AGENT,
        *,
        timeout: float = 10.0,
        stats: CrawlStats | None = None,
        log: CrawlLog | None = None,
        sleep: Sleep | None = None,
        jitter: Jitter | None = None,
        base: float = 0.5,
        cap: float = 8.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._stats = stats
        self._log = log
        self._sleep = asyncio.sleep if sleep is None else sleep
        self._jitter = jitter
        self._base = base
        self._cap = cap
        self._cache: dict[str, RobotFileParser] = {}
        self._loading: dict[str, asyncio.Task[RobotFileParser]] = {}
        self._failed: set[str] = set()

    def loaded(self, url: str) -> bool:
        """True when this origin's ``robots.txt`` is already in the cache."""
        return origin_of(url) in self._cache

    def failed(self, url: str) -> bool:
        return origin_of(url) in self._failed

    async def allowed(self, url: str) -> bool:
        parser = await self._parser_for(origin_of(url))
        return bool(parser.can_fetch(self._user_agent, url))

    async def _parser_for(self, origin: str) -> RobotFileParser:
        cached = self._cache.get(origin)
        if cached is not None:
            return cached
        # No await between the lookup and the insert, so two callers share one task.
        task = self._loading.get(origin)
        if task is None:
            task = asyncio.create_task(self._load(origin))
            self._loading[origin] = task
        parser = await task
        self._cache[origin] = parser
        return parser

    async def _load(self, origin: str) -> RobotFileParser:
        robots_url = f"{origin}/robots.txt"
        result = await fetch(
            self._client,
            robots_url,
            timeout=self._timeout,
            max_attempts=3,
            base=self._base,
            cap=self._cap,
            sleep=self._sleep,
            jitter=self._jitter,
            stats=self._stats,
        )
        if self._log is not None:
            for number, record in enumerate(result.attempts, start=1):
                self._log.attempt(robots_url, number, record)

        parser = RobotFileParser()
        if result.status == 404:
            parser.parse(["User-agent: *", "Allow: /"])
            return parser
        if not result.ok:
            logger.info(
                "robots.txt unavailable for %s (%s); disallowing",
                origin,
                result.error,
            )
            self._failed.add(origin)
            parser.parse(["User-agent: *", "Disallow: /"])
            return parser
        text = result.body.decode("utf-8", errors="replace")
        parser.parse(text.splitlines())
        return parser

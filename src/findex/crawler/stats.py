"""Live counters for the crawl display. Mutated only on the event-loop thread."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class CrawlStats:
    """Pages written, failures, requests waiting on the network, frontier depth."""

    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    started: float = field(default_factory=time.perf_counter)
    frontier: asyncio.Queue[str] | None = None

    @property
    def queue_depth(self) -> int:
        if self.frontier is None:
            return 0
        return self.frontier.qsize()

    @property
    def pages_per_sec(self) -> float:
        elapsed = time.perf_counter() - self.started
        if elapsed <= 0:
            return 0.0
        return self.pages / elapsed

    def note_request_started(self) -> None:
        self.in_flight += 1

    def note_request_finished(self) -> None:
        if self.in_flight > 0:
            self.in_flight -= 1

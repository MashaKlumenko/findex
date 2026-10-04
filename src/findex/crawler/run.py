"""Stream pages to JSON lines and paint live counters. Indexing stays outside."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler.crawl import crawl
from findex.crawler.fetch import USER_AGENT
from findex.crawler.log import CrawlLog
from findex.crawler.stats import CrawlStats


def render_stats(stats: CrawlStats) -> Table:
    """The five numbers the lab asks to keep on screen."""
    table = Table(title="findex crawl")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="cyan")
    table.add_row("Pages", str(stats.pages))
    table.add_row("Pages/s", f"{stats.pages_per_sec:.2f}")
    table.add_row("In flight", str(stats.in_flight))
    table.add_row("Errors", str(stats.errors))
    table.add_row("Queue", str(stats.queue_depth))
    return table


async def _paint(live: Live, stats: CrawlStats, stop: asyncio.Event) -> None:
    while not stop.is_set():
        live.update(render_stats(stats))
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            continue
    live.update(render_stats(stats))


async def run_crawl(
    seeds: Sequence[str],
    *,
    out: Path,
    log_path: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    min_delay: float,
    timeout: float,
    allowed_domains: list[str] | None,
    console: Console,
    progress: bool,
    user_agent: str = USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CrawlStats:
    """Write one JSON object per page as it arrives. The file is flushed each line."""
    stats = CrawlStats()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = CrawlLog(log_path)
    stop = asyncio.Event()
    try:
        with out.open("w", encoding="utf-8") as handle:
            live_cm = (
                Live(render_stats(stats), console=console, refresh_per_second=4)
                if progress
                else nullcontext(None)
            )
            with live_cm as live:
                painter: asyncio.Task[None] | None = None
                if live is not None:
                    painter = asyncio.create_task(_paint(live, stats, stop))
                try:
                    async for page in crawl(
                        list(seeds),
                        max_pages=max_pages,
                        concurrency=concurrency,
                        per_host=per_host,
                        allowed_domains=allowed_domains,
                        min_delay=min_delay,
                        timeout=timeout,
                        user_agent=user_agent,
                        stats=stats,
                        log=log,
                        transport=transport,
                    ):
                        record = {
                            "id": page.url,
                            "url": page.url,
                            "title": page.title,
                            "text": page.text,
                            "fetched_at": page.fetched_at,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                finally:
                    stop.set()
                    if painter is not None:
                        await painter
    finally:
        log.close()
    return stats

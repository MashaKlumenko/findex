"""Benchmark the async crawler against the local mirror.

Same seed, same max_pages, concurrency in {1, 5, 20}. The mirror waits on
purpose so the client is idle on the socket — the case asyncio is for.

    python scripts/lab06_bench.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.crawler.mirror import serve_in_thread  # noqa: E402
from findex.crawler.run import run_crawl  # noqa: E402
from findex.resources import ResourceMonitor  # noqa: E402
from findex.stats import format_bytes  # noqa: E402

MAX_PAGES = 200
LATENCY = 0.08
OUT_DIR = ROOT / "data" / "lab06"
DOCS = ROOT / "docs"


def _crawl(
    seed: str,
    *,
    concurrency: int,
    per_host: int,
    delay: float,
    max_pages: int,
    out: Path,
    log_path: Path,
    debug: bool = False,
) -> dict[str, object]:
    from rich.console import Console

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    async def run():
        return await run_crawl(
            [seed],
            out=out,
            log_path=log_path,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=delay,
            timeout=10,
            allowed_domains=None,
            console=Console(quiet=True),
            progress=False,
        )

    stats = asyncio.run(run(), debug=debug)
    wall = time.perf_counter() - started
    _cpu, peak = monitor.finish(children=False)
    return {
        "concurrency": concurrency,
        "per_host": per_host,
        "delay": delay,
        "pages": stats.pages,
        "errors": stats.errors,
        "wall_seconds": round(wall, 3),
        "pages_per_sec": round(stats.pages / wall, 2) if wall else 0,
        "peak_rss": peak,
        "peak_rss_human": format_bytes(peak),
    }


def _debug_clean(seed: str, out: Path, log_path: Path) -> bool:
    buffer: list[str] = []

    class _Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            buffer.append(record.getMessage())

    handler = _Grab()
    handler.setLevel(logging.DEBUG)
    loop_log = logging.getLogger("asyncio")
    previous = loop_log.level
    loop_log.setLevel(logging.DEBUG)
    loop_log.addHandler(handler)
    try:
        _crawl(
            seed,
            concurrency=10,
            per_host=10,
            delay=0,
            max_pages=30,
            out=out,
            log_path=log_path,
            debug=True,
        )
    finally:
        loop_log.removeHandler(handler)
        loop_log.setLevel(previous)
    text = "\n".join(buffer)
    return "Executing" not in text and "took" not in text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    server, _thread = serve_in_thread(latency=LATENCY)
    host, port = server.server_address
    seed = f"http://{host}:{port}/page/0"
    flaky = f"http://{host}:{port}/flaky"
    try:
        rows = []
        for concurrency in (1, 5, 20):
            row = _crawl(
                seed,
                concurrency=concurrency,
                per_host=concurrency,
                delay=0,
                max_pages=MAX_PAGES,
                out=OUT_DIR / f"crawl-{concurrency}.jsonl",
                log_path=OUT_DIR / f"crawl-{concurrency}.log",
            )
            rows.append(row)
            print(
                f"concurrency={concurrency} pages={row['pages']} "
                f"wall={row['wall_seconds']}s {row['pages_per_sec']}/s "
                f"errors={row['errors']} rss={row['peak_rss_human']}"
            )
        limited = _crawl(
            seed,
            concurrency=20,
            per_host=2,
            delay=0.05,
            max_pages=MAX_PAGES,
            out=OUT_DIR / "crawl-host-limit.jsonl",
            log_path=OUT_DIR / "crawl-host-limit.log",
        )
        print(
            "per_host=2 delay=0.05s "
            f"wall={limited['wall_seconds']}s {limited['pages_per_sec']}/s"
        )
        failure = _crawl(
            flaky,
            concurrency=1,
            per_host=1,
            delay=0,
            max_pages=1,
            out=OUT_DIR / "flaky.jsonl",
            log_path=OUT_DIR / "flaky.log",
        )
        failure_log = (OUT_DIR / "flaky.log").read_text(encoding="utf-8")
        (DOCS / "lab06-failure.log").write_text(failure_log, encoding="utf-8")
        debug_ok = _debug_clean(
            seed, OUT_DIR / "debug.jsonl", OUT_DIR / "debug.log"
        )
    finally:
        server.shutdown()

    payload = {
        "machine": platform.platform(),
        "python": platform.python_version(),
        "seed_path": "/page/0",
        "latency_seconds": LATENCY,
        "max_pages": MAX_PAGES,
        "note": (
            "per_host equals concurrency and delay is 0 so the table measures "
            "in-flight requests. The host-limit row keeps per_host=2."
        ),
        "rows": rows,
        "host_limit": limited,
        "failure": failure,
        "failure_log": failure_log,
        "debug_clean": debug_ok,
    }
    (DOCS / "lab06-bench.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("debug_clean", debug_ok)
    print("wrote", DOCS / "lab06-bench.json")


if __name__ == "__main__":
    main()

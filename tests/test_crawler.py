"""Crawler tests. HTTP goes through MockTransport or an in-process transport."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from email.utils import formatdate
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler.crawl import crawl
from findex.crawler.fetch import (
    USER_AGENT,
    backoff_seconds,
    fetch,
    parse_retry_after,
)
from findex.crawler.log import format_log_line
from findex.crawler.mirror import serve_in_thread
from findex.crawler.parse import parse_html
from findex.crawler.robots import RobotsCache
from findex.crawler.run import render_stats, run_crawl
from findex.crawler.urls import canonicalize

runner = CliRunner()


def _html(title: str, text: str, hrefs: list[str]) -> str:
    links = "".join(f'<a href="{href}">x</a>' for href in hrefs)
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><p>{text}</p>{links}</body></html>"
    )


def test_canonicalize_scheme_host_fragment_slash_and_query() -> None:
    assert canonicalize("HTTP://Example.COM/Docs/") == "http://example.com/Docs"
    assert canonicalize("http://example.com/a#section") == "http://example.com/a"
    assert canonicalize("http://example.com:80/a/") == "http://example.com/a"
    assert canonicalize("https://example.com:443/a") == "https://example.com/a"
    assert canonicalize("http://example.com/a?b=1&a=2") == "http://example.com/a?a=2&b=1"
    assert canonicalize("http://example.com/a/../b") == "http://example.com/b"
    assert canonicalize("http://example.com") == "http://example.com/"


def test_parse_html_skips_script_and_keeps_links() -> None:
    html = _html(
        "Hello",
        "Visible",
        ["/next", "http://example.com/a#x", "mailto:me@example.com"],
    )
    html = html.replace(
        "<p>Visible</p>",
        "<p>Visible</p><script>secret token</script>",
    )
    title, text, links = parse_html("http://example.com/start", html)
    assert title == "Hello"
    assert text.startswith("Hello\n")
    assert "Visible" in text
    assert "secret" not in text
    assert "http://example.com/next" in links
    assert "http://example.com/a" in links
    assert all(not link.startswith("mailto:") for link in links)


def test_backoff_grows_and_honors_retry_after() -> None:
    first = backoff_seconds(0, None, base=0.5, cap=8.0, jitter=0.0)
    doubled = backoff_seconds(2, None, base=0.5, cap=8.0, jitter=0.25)
    capped = backoff_seconds(10, None, base=0.5, cap=8.0, jitter=0.0)
    honored = backoff_seconds(0, 3.0, base=0.5, cap=8.0, jitter=0.0)
    assert first == pytest.approx(0.5)
    assert doubled == pytest.approx(2.25)
    assert capped == pytest.approx(8.0)
    assert honored == pytest.approx(3.0)


def test_parse_retry_after_seconds_and_http_date() -> None:
    assert parse_retry_after("2") == pytest.approx(2.0)
    stamp = formatdate(time.time() + 30, usegmt=True)
    assert parse_retry_after(stamp) >= 20
    assert parse_retry_after("not-a-date") is None


def test_log_line_includes_retry_after() -> None:
    line = format_log_line(
        ts="2026-10-04T12:00:00Z",
        url="http://example.com/a",
        status=429,
        nbytes=9,
        elapsed=0.012,
        attempt=1,
        error="http_429",
        retry_after=1.0,
    )
    assert "status=429" in line
    assert "bytes=9" in line
    assert "error=http_429" in line
    assert "retry_after=1.000" in line


async def _asleep(_: float) -> None:
    return None


def _client(transport: httpx.BaseTransport, **kwargs: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport,
        follow_redirects=True,
        max_redirects=5,
        timeout=None,
        **kwargs,  # type: ignore[arg-type]
    )


def test_fetch_records_status_bytes_and_does_not_retry_404() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["user-agent"] == USER_AGENT
        return httpx.Response(404, content=b"missing")

    async def run() -> None:
        async with _client(
            httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT}
        ) as client:
            result = await fetch(client, "http://example.test/missing", sleep=_asleep)
        assert result.status == 404
        assert result.body == b"missing"
        assert result.elapsed >= 0
        assert len(result.attempts) == 1
        assert calls["n"] == 1

    asyncio.run(run())


def test_fetch_retries_500_then_succeeds_and_stops_after_three() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, content=b"no")
        return httpx.Response(200, content=b"ok")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with _client(transport) as client:
            result = await fetch(
                client,
                "http://example.test/ok",
                sleep=_asleep,
                jitter=lambda: 0.0,
                base=0.0,
            )
        assert result.ok
        assert result.body == b"ok"
        assert len(result.attempts) == 3
        assert calls["n"] == 3

        calls["n"] = 0

        def always_500(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, content=b"down")

        async with _client(httpx.MockTransport(always_500)) as client:
            failed = await fetch(
                client,
                "http://example.test/down",
                sleep=_asleep,
                jitter=lambda: 0.0,
                base=0.0,
            )
        assert failed.status == 503
        assert failed.error == "http_503"
        assert len(failed.attempts) == 3
        assert calls["n"] == 3

    asyncio.run(run())


def test_fetch_honors_retry_after_on_429() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, content=b"later", headers={"Retry-After": "2"}
            )
        return httpx.Response(200, content=b"ok")

    async def record(delay: float) -> None:
        slept.append(delay)

    async def run() -> None:
        async with _client(httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "http://example.test/limited",
                sleep=record,
                jitter=lambda: 0.0,
                base=0.5,
            )
        assert result.ok
        assert slept == pytest.approx([2.0])
        assert result.attempts[0].retry_after == pytest.approx(2.0)
        assert calls["n"] == 2

    asyncio.run(run())


def test_fetch_retries_timeout_and_not_connection_errors() -> None:
    class Hang(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            await asyncio.sleep(1)
            return httpx.Response(200, content=b"late")

    hang = Hang()

    async def run_timeout() -> None:
        async with _client(hang) as client:
            result = await fetch(
                client,
                "http://example.test/slow",
                timeout=0.05,
                max_attempts=3,
                sleep=_asleep,
                jitter=lambda: 0.0,
                base=0.0,
            )
        assert result.error == "timeout"
        assert len(result.attempts) == 3
        assert hang.calls == 3

    asyncio.run(run_timeout())

    calls = {"n": 0}

    def boom(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    async def run_connect() -> None:
        async with _client(httpx.MockTransport(boom)) as client:
            result = await fetch(client, "http://example.test/x", sleep=_asleep)
        assert result.error == "ConnectError"
        assert len(result.attempts) == 1
        assert calls["n"] == 1

    asyncio.run(run_connect())


def test_fetch_does_not_retry_redirect_loops() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": str(request.url)})

    async def run() -> None:
        async with _client(httpx.MockTransport(handler)) as client:
            result = await fetch(client, "http://example.test/loop", sleep=_asleep)
        assert result.error == "redirect_loop"
        assert len(result.attempts) == 1

    asyncio.run(run())


def test_robots_cache_per_host_and_fail_closed() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.host}{request.url.path}")
        host = request.url.host
        if host == "good.test":
            body = b"User-agent: *\nDisallow: /secret\nAllow: /\n"
            return httpx.Response(
                200, content=body, headers={"content-type": "text/plain"}
            )
        if host == "missing.test":
            return httpx.Response(404, content=b"")
        return httpx.Response(500, content=b"no")

    async def run() -> None:
        async with _client(httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client, sleep=_asleep, jitter=lambda: 0.0, base=0.0)
            assert await cache.allowed("http://good.test/docs")
            assert not await cache.allowed("http://good.test/secret")
            assert await cache.allowed("http://good.test/other")
            assert await cache.allowed("http://missing.test/anywhere")
            assert not await cache.allowed("http://down.test/a")
            assert cache.failed("http://down.test/a")
        robots_gets = [item for item in calls if item.endswith("/robots.txt")]
        assert robots_gets.count("good.test/robots.txt") == 1
        assert robots_gets.count("down.test/robots.txt") == 3

    asyncio.run(run())


def _site(
    pages: dict[str, str], robots: bytes | None = None
) -> tuple[list[str], httpx.MockTransport]:
    calls: list[str] = []
    rules = robots if robots is not None else b"User-agent: *\nAllow: /\n"

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200, content=rules, headers={"content-type": "text/plain"}
            )
        body = pages.get(request.url.path)
        if body is None:
            return httpx.Response(
                404, content=b"missing", headers={"content-type": "text/plain"}
            )
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/html"}
        )

    return calls, httpx.MockTransport(handler)


def test_crawl_canonicalizes_obeys_robots_and_stops_at_max_pages() -> None:
    pages = {
        "/a": _html(
            "Alpha",
            "alpha body",
            [
                "http://EXAMPLE.test/a#frag",
                "http://example.test/a/",
                "/b?b=1&a=2",
                "/c",
                "/secret",
                "http://evil.test/no",
                "/d",
            ],
        ),
        "/b": _html("Beta", "beta body", []),
        "/c": _html("Cee", "cee body", []),
        "/d": _html("Dee", "dee body", []),
        "/secret": _html("Secret", "nope", []),
    }
    calls, transport = _site(
        pages, robots=b"User-agent: *\nDisallow: /secret\nAllow: /\n"
    )

    async def run() -> list[str]:
        found: list[str] = []
        async with asyncio.timeout(5):
            async for page in crawl(
                ["http://EXAMPLE.test/a#top"],
                max_pages=4,
                concurrency=4,
                per_host=2,
                min_delay=0,
                transport=transport,
                backoff_base=0,
                jitter=lambda: 0.0,
                sleep=_asleep,
            ):
                found.append(page.url)
        return found

    urls = asyncio.run(run())
    # /secret occupies a slot and is not fetched. /d is past max_pages.
    assert urls[0] == "http://example.test/a"
    assert set(urls) == {
        "http://example.test/a",
        "http://example.test/b?a=2&b=1",
        "http://example.test/c",
    }
    assert calls.count("/a") == 1
    assert "/secret" not in calls
    assert "/d" not in calls
    assert "/no" not in calls


def test_crawl_max_pages_cancels_and_closes_the_client() -> None:
    pages = {
        "/page/0": _html("P0", "start", [f"/page/{i}" for i in range(1, 30)]),
    }
    pages.update(
        {f"/page/{i}": _html(f"P{i}", f"body {i}", []) for i in range(1, 30)}
    )

    class Closing(httpx.MockTransport):
        def __init__(self) -> None:
            super().__init__(self._handle)
            self.closed = False
            self.calls: list[str] = []

        def _handle(self, request: httpx.Request) -> httpx.Response:
            self.calls.append(request.url.path)
            if request.url.path == "/robots.txt":
                return httpx.Response(
                    200,
                    content=b"User-agent: *\nAllow: /\n",
                    headers={"content-type": "text/plain"},
                )
            body = pages.get(request.url.path, "<html><title>x</title><p>x</p></html>")
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/html"}
            )

        async def aclose(self) -> None:
            self.closed = True
            await super().aclose()

    transport = Closing()

    async def run() -> int:
        n = 0
        async with asyncio.timeout(5):
            async for _page in crawl(
                ["http://example.test/page/0"],
                max_pages=4,
                concurrency=4,
                per_host=4,
                min_delay=0,
                transport=transport,
                sleep=_asleep,
                jitter=lambda: 0.0,
                backoff_base=0,
            ):
                n += 1
                if n == 2:
                    break
        return n

    assert asyncio.run(run()) == 2
    assert transport.closed
    page_calls = [path for path in transport.calls if path.startswith("/page/")]
    assert len(page_calls) <= 4


def test_per_host_delay_and_semaphore() -> None:
    times: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/robots.txt":
            times.append(time.perf_counter())
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                content=b"User-agent: *\nAllow: /\n",
                headers={"content-type": "text/plain"},
            )
        links = ["/b", "/c", "/d"] if request.url.path == "/a" else []
        body = _html(request.url.path, "text", links)
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/html"}
        )

    async def run() -> None:
        async with asyncio.timeout(5):
            async for _page in crawl(
                ["http://example.test/a"],
                max_pages=4,
                concurrency=4,
                per_host=1,
                min_delay=0.05,
                transport=httpx.MockTransport(handler),
                sleep=asyncio.sleep,
                jitter=lambda: 0.0,
                backoff_base=0,
            ):
                pass

    asyncio.run(run())
    assert len(times) == 4
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert all(gap >= 0.04 for gap in gaps)

    class Slow(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.current = 0
            self.peak = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.current += 1
            self.peak = max(self.peak, self.current)
            try:
                await asyncio.sleep(0.03)
            finally:
                self.current -= 1
            if request.url.path == "/robots.txt":
                return httpx.Response(
                    200,
                    content=b"User-agent: *\nAllow: /\n",
                    headers={"content-type": "text/plain"},
                )
            name = request.url.path
            links = [f"/p/{i}" for i in range(1, 12)] if name == "/p/0" else []
            body = _html(name, "body", links)
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/html"}
            )

    slow = Slow()

    async def run_limited() -> None:
        async with asyncio.timeout(5):
            async for _page in crawl(
                ["http://example.test/p/0"],
                max_pages=8,
                concurrency=8,
                per_host=2,
                min_delay=0,
                transport=slow,
                sleep=_asleep,
                jitter=lambda: 0.0,
                backoff_base=0,
            ):
                pass

    asyncio.run(run_limited())
    assert slow.peak <= 2
    assert slow.peak >= 2


def test_run_crawl_streams_jsonl_that_iter_documents_reads(tmp_path: Path) -> None:
    pages = {
        "/": _html("Home", "cooperative scheduling of coroutines", ["/a"]),
        "/a": _html("Other", "more text", []),
    }
    _calls, transport = _site(pages)
    out = tmp_path / "crawl.jsonl"
    log_path = tmp_path / "crawl.log"

    async def run() -> None:
        await run_crawl(
            ["http://example.test/"],
            out=out,
            log_path=log_path,
            max_pages=2,
            concurrency=2,
            per_host=2,
            min_delay=0,
            timeout=5,
            allowed_domains=None,
            console=Console(file=io.StringIO(), force_terminal=True),
            progress=True,
            transport=transport,
        )

    asyncio.run(run())
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    docs = list(iter_documents(out))
    assert len(docs) == 2
    assert any("cooperative" in doc.text for doc in docs)
    log_text = log_path.read_text(encoding="utf-8")
    assert "status=200" in log_text
    assert "robots.txt" in log_text
    table = render_stats.__doc__ or ""
    assert "five" in table


def test_debug_mode_does_not_report_a_blocked_loop() -> None:
    pages = {"/": _html("Home", "text", [f"/p{i}" for i in range(6)])}
    pages.update({f"/p{i}": _html(f"P{i}", f"body {i}", []) for i in range(6)})
    _calls, transport = _site(pages)
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    loop_log = logging.getLogger("asyncio")
    previous = loop_log.level
    loop_log.setLevel(logging.DEBUG)
    loop_log.addHandler(handler)

    async def run() -> None:
        async for _page in crawl(
            ["http://example.test/"],
            max_pages=7,
            concurrency=4,
            per_host=4,
            min_delay=0,
            transport=transport,
            sleep=_asleep,
            jitter=lambda: 0.0,
            backoff_base=0,
        ):
            pass

    try:
        asyncio.run(run(), debug=True)
    finally:
        loop_log.removeHandler(handler)
        loop_log.setLevel(previous)
    assert "Executing" not in buffer.getvalue()


def test_cli_crawl_against_local_mirror(tmp_path: Path) -> None:
    server, _thread = serve_in_thread(latency=0)
    port = server.server_address[1]
    out = tmp_path / "crawl.jsonl"
    log_path = tmp_path / "crawl.log"
    try:
        result = runner.invoke(
            app,
            [
                "crawl",
                f"http://127.0.0.1:{port}/page/0",
                "--max-pages",
                "5",
                "--concurrency",
                "4",
                "--per-host",
                "4",
                "--delay",
                "0",
                "--timeout",
                "5",
                "--out",
                str(out),
                "--log",
                str(log_path),
                "--quiet",
            ],
        )
    finally:
        server.shutdown()
    assert result.exit_code == 0, result.output
    docs = list(iter_documents(out))
    assert len(docs) == 5
    assert "Pages" in result.output or "pages" in result.output.lower()

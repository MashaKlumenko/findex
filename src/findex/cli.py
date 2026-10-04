"""Command-line interface for findex using Typer and Rich."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

# pyright: reportUnusedImport=false, reportUnknownVariableType=false
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from findex.crawler.run import render_stats, run_crawl
from findex.index import DEFAULT_EXECUTOR, EXECUTORS, build_index_parallel
from findex.rank import get_scorer, ranked_search
from findex.search import boolean_search
from findex.stats import format_bytes
from findex.store import open_index, save
from findex.tokenize import tokenize as tokenizer

__all__ = ["app", "tokenizer"]  # noqa: F401  re-export for tests

# Створюємо аплікацію Typer
app = typer.Typer(help="findex: CLI search engine tool.")

# Розділяємо вивід: стандартний console йде в stdout, помилки — в stderr
console = Console()
error_console = Console(stderr=True)
logger = logging.getLogger("findex")


def setup_logging(verbose: int) -> None:
    """Configure structured logging to stderr based on verbosity level."""
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    ),
) -> None:
    """Global configuration for all findex subcommands."""
    setup_logging(verbose)


@app.command(name="index")
def index_command(
    corpus: Path = typer.Argument(..., help="Path to the document corpus dir/file."),
    output: Path = typer.Option(
        Path("index.json"), "--out", "-o", help="Output index path."
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-j",
        help="Number of chunks. Default: 1 for serial, os.cpu_count() otherwise.",
    ),
    executor: Literal["serial", "threads", "processes"] = typer.Option(
        DEFAULT_EXECUTOR,
        "--executor",
        help="serial loop, ThreadPoolExecutor, or ProcessPoolExecutor.",
    ),
    positions: bool = typer.Option(False, "--positions", help="Store token offsets."),
    limit: int | None = typer.Option(
        None, "--limit", help="Index only the first N documents."
    ),
) -> None:
    """Build an inverted index from a document corpus."""
    if executor not in EXECUTORS:
        error_console.print(f"Error: unknown executor {executor!r}")
        raise typer.Exit(code=1)
    logger.info("Starting index construction for: %s (%s)", corpus, executor)
    try:
        index, report = build_index_parallel(
            corpus,
            positions=positions,
            limit=limit,
            workers=workers,
            executor=executor,
        )
        save(index, output)
    except Exception:
        # The worker stack is on ``__cause__`` for process pools. Print it;
        # do not swallow the failure into a hang or a one-line message.
        error_console.print(traceback.format_exc())
        raise typer.Exit(code=1)

    gil = report.gil_enabled
    gil_text = "n/a" if gil is None else str(gil)
    table = Table(title="Index build")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="light_blue")
    table.add_row("Documents", str(index.num_docs))
    table.add_row("Vocabulary", str(len(index)))
    table.add_row("Executor", report.executor)
    table.add_row("Workers", str(report.workers))
    table.add_row("Wall", f"{report.wall_seconds:.3f} s")
    table.add_row("CPU", f"{report.cpu_seconds:.3f} s")
    table.add_row("Peak RSS (parent)", format_bytes(report.peak_rss))
    table.add_row("Merge", f"{report.merge_seconds:.3f} s")
    table.add_row("sys._is_gil_enabled()", gil_text)
    table.add_row("Wrote", str(output))
    console.print(table)


@app.command(name="search")
def search_command(
    index_path: Path = typer.Argument(..., help="Path to the saved index file."),
    query: str = typer.Argument(..., help="Search query string."),
    boolean: bool = typer.Option(False, "--boolean", "-b", help="Use Boolean search instead of ranked."),
    engine: Literal["merge", "set"] = typer.Option("merge", "--engine", help="Boolean engine type."),
    scorer_type: Literal["bm25", "tfidf"] = typer.Option("bm25", "--scorer", help="Ranked scorer function."),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum hits to return."),
    json_mode: bool = typer.Option(False, "--json", help="Output raw JSON data directly to stdout."),
) -> None:
    """Query the index and display matching documents."""
    if not index_path.exists():
        error_console.print(f"Error: Index file not found at '{index_path}'")
        raise typer.Exit(code=1)

    try:
        with open_index(index_path) as index:
            if boolean:
                hits = boolean_search(index, query, engine=engine)
                results_json = [{"doc_id": doc_id} for doc_id in hits[:limit]]

                if json_mode:
                    sys.stdout.write(json.dumps(results_json) + "\n")
                    return

                # Rich-таблиця для Boolean результатів
                table = Table(title=f"Boolean Hits for: '{query}' (Engine: {engine})")
                table.add_column("Document ID", style="cyan")
                for doc_id in hits[:limit]:
                    table.add_row(str(doc_id))
                console.print(table)
            else:
                scorer = get_scorer(scorer_type)
                results = ranked_search(index, query, scorer=scorer, k=limit)
                results_json = [
                    {"doc_id": r.doc_id, "score": r.score, "title": r.title, "snippet": r.snippet}
                    for r in results
                ]

                if json_mode:
                    sys.stdout.write(json.dumps(results_json) + "\n")
                    return

                # Rich-таблиця для Ranked результатів (обов'язково для захисту!)
                table = Table(title=f"Ranked Search Results for: '{query}'")
                table.add_column("Score", justify="right", style="magenta")
                table.add_column("Doc ID", justify="center", style="cyan")
                table.add_column("Title", style="green")
                table.add_column("Snippet", style="white")

                for r in results:
                    table.add_row(f"{r.score:8.4f}", str(r.doc_id), r.title, r.snippet)
                console.print(table)

    except Exception as e:
        error_console.print(f"Error during search: {e}")
        raise typer.Exit(code=1)


@app.command(name="stats")
def stats_command(
    index_path: Path = typer.Argument(..., help="Path to the saved index file."),
) -> None:
    """Display structural statistics of the inverted index."""
    if not index_path.exists():
        error_console.print(f"Error: Index file not found at '{index_path}'")
        raise typer.Exit(code=1)

    try:
        with open_index(index_path) as index:
            table = Table(title="Index Metadata Statistics")
            table.add_column("Metric", style="bold yellow")
            table.add_column("Value", style="light_blue")
            table.add_row("Total Documents", str(index.num_docs))
            table.add_row("Vocabulary Size", str(len(index)))
            if hasattr(index, 'avg_doc_length'):
                table.add_row("Avg Doc Length", f"{index.avg_doc_length:.2f}")
            console.print(table)
    except Exception as e:
        error_console.print(f"Error reading stats: {e}")
        raise typer.Exit(code=1)


@app.command(name="crawl")
def crawl_command(
    url: str = typer.Argument(
        ...,
        help="Seed URL. Only this host is followed unless --allow-domain is set.",
    ),
    max_pages: int = typer.Option(500, "--max-pages", help="Stop after N pages."),
    concurrency: int = typer.Option(
        10, "--concurrency", help="Workers and the global cap."
    ),
    per_host: int = typer.Option(2, "--per-host", help="Cap for one host."),
    delay: float = typer.Option(0.2, "--delay", help="Gap between hits to one host."),
    out: Path = typer.Option(
        Path("data/crawl.jsonl"), "--out", help="JSONL output."
    ),
    log_path: Path = typer.Option(Path("crawl.log"), "--log", help="Status log."),
    timeout: float = typer.Option(10.0, "--timeout", help="Request timeout."),
    allow_domain: list[str] = typer.Option(
        [], "--allow-domain", help="Extra host. Repeatable."
    ),
    debug: bool = typer.Option(False, "--debug", help="Warn if the loop blocks."),
    quiet: bool = typer.Option(False, "--quiet", help="Skip the live counters."),
) -> None:
    """Fetch pages concurrently and stream them as JSON lines.

    Indexing stays a separate ``findex index`` run. Tokenizing inside the
    event loop would stall every in-flight request.
    """
    progress = not quiet and console.is_terminal

    async def _go():
        return await run_crawl(
            [url],
            out=out,
            log_path=log_path,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=delay,
            timeout=timeout,
            allowed_domains=allow_domain or None,
            console=console,
            progress=progress,
        )

    started = time.perf_counter()
    try:
        stats = asyncio.run(_go(), debug=debug)
    except KeyboardInterrupt:
        error_console.print("crawl interrupted")
        raise typer.Exit(code=130) from None
    except Exception:
        error_console.print(traceback.format_exc())
        raise typer.Exit(code=1) from None

    wall = time.perf_counter() - started
    summary = render_stats(stats)
    summary.title = "Crawl finished"
    summary.add_row("Wall", f"{wall:.3f} s")
    summary.add_row("Wrote", str(out))
    summary.add_row("Log", str(log_path))
    console.print(summary)


@app.command(name="serve")
def serve_command(
    port: int = typer.Option(8000, "--port", "-p", help="TCP port."),
    workers: int = typer.Option(
        1, "--workers", "-w", min=1, help="Uvicorn worker processes."
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    index: Path | None = typer.Option(
        None,
        "--index",
        "-i",
        help="Index file. Exported as INDEX_PATH before workers start.",
    ),
) -> None:
    """Serve the search API and web page.

    The index path is configuration, so it lives in the environment.
    ``--index`` only writes ``INDEX_PATH`` for this process and its workers.
    """
    if index is not None:
        os.environ["INDEX_PATH"] = str(index.resolve())
    import uvicorn

    uvicorn.run(
        "findex.web:app",
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()

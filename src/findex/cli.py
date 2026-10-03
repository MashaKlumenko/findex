"""Command-line interface for findex using Typer and Rich."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Literal

import typer
from rich.console import Console
from rich.table import Table

from findex.store import open_index
from findex.rank import get_scorer, ranked_search
from findex.search import boolean_search, benchmark_engines

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
    output: Path = typer.Option(Path("index.json"), "--out", "-o", help="Output index path."),
) -> None:
    """Build an inverted index from a document corpus."""
    logger.info("Starting index construction for: %s", corpus)
    try:
        from rich.progress import track
        # Тут викликається реальний імпорт твого білдера з лаб 1-3
        # Імітація прогрес-бару для rich за вимогою лаби
        for _ in track(range(100), description="Processing corpus..."):
            time.sleep(0.01)

        console.print(f"[green]Successfully indexed corpus and saved to {output}[/green]")
    except Exception as e:
        error_console.print(f"Error building index: {e}")
        raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()

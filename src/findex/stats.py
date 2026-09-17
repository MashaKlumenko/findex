"""Single-pass corpus statistics over a lazy document/token pipeline.

    python -m findex.stats data/
    python -m findex.stats data/ --limit 100
    python -m findex.stats data/ --eager

Peak memory is traced with ``tracemalloc``; wall time with ``perf_counter``.
The only structure that is *supposed* to grow is the term ``Counter``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

TOP_N = 50


def collect_stats(docs: Iterable[Document]) -> tuple[int, int, Counter[str]]:
    """Return ``(n_docs, n_tokens, term_counts)`` in one pass."""
    n_docs = 0
    n_tokens = 0
    counts: Counter[str] = Counter()
    for doc in docs:
        n_docs += 1
        for tok in tokenize(doc.text):
            n_tokens += 1
            counts[tok] += 1
    return n_docs, n_tokens, counts


def eager_documents(root: Path, limit: int | None) -> list[Document]:
    """The version we are *not* supposed to write: lists all the way down."""
    docs = list(iter_documents(root))
    if limit is not None:
        docs = docs[:limit]
    return docs


def eager_stats(docs: list[Document]) -> tuple[int, int, Counter[str]]:
    token_lists = [list(tokenize(doc.text)) for doc in docs]
    counts: Counter[str] = Counter()
    n_tokens = 0
    for tokens in token_lists:
        n_tokens += len(tokens)
        counts.update(tokens)
    return len(docs), n_tokens, counts


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def print_report(
    n_docs: int,
    n_tokens: int,
    counts: Counter[str],
    elapsed: float,
    peak: int,
    *,
    mode: str,
) -> None:
    vocab = len(counts)
    print(f"mode:           {mode}")
    print(f"documents:      {n_docs}")
    print(f"tokens:         {n_tokens}")
    print(f"vocabulary:     {vocab}")
    print(f"elapsed:        {elapsed:.3f} s")
    print(f"peak memory:    {format_bytes(peak)} ({peak} bytes)")
    print()
    print(f"top {TOP_N} terms:")
    for rank, (term, freq) in enumerate(counts.most_common(TOP_N), start=1):
        print(f"  {rank:2d}. {term:20s} {freq}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream a corpus and print term statistics.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="directory of .txt/.md/.jsonl files, or a single .jsonl file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only the first N documents (itertools.islice on the lazy stream)",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="deliberately materialize lists (for the Lab 1 memory table)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log skipped files and decode replacements",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tracemalloc.start()
    t0 = time.perf_counter()

    if args.eager:
        docs = eager_documents(args.root, args.limit)
        n_docs, n_tokens, counts = eager_stats(docs)
        mode = "eager (lists)"
    else:
        docs_iter: Iterable[Document] = iter_documents(args.root)
        if args.limit is not None:
            docs_iter = islice(docs_iter, args.limit)
        n_docs, n_tokens, counts = collect_stats(docs_iter)
        mode = "lazy (generators)"

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print_report(n_docs, n_tokens, counts, elapsed, peak, mode=mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

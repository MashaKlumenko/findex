from __future__ import annotations

import argparse
import logging
import sys
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path

from findex.index import Index
from findex.stats import format_bytes
from findex.store import load
from findex.tokenize import tokenize

# Токенізатор з Лаби 1 робить lowercase, тому оператори мають бути в нижньому регістрі
OPERATORS = frozenset({"and", "or", "not"})


def merge_and(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Two-pointer intersection of sorted doc-id lists."""
    i = j = 0
    out: list[int] = []
    n_left, n_right = len(left), len(right)
    while i < n_left and j < n_right:
        a, b = left[i], right[j]
        if a == b:
            out.append(a)
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return out


def merge_or(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Two-pointer union of sorted doc-id lists."""
    i = j = 0
    out: list[int] = []
    n_left, n_right = len(left), len(right)
    while i < n_left and j < n_right:
        a, b = left[i], right[j]
        if a == b:
            out.append(a)
            i += 1
            j += 1
        elif a < b:
            out.append(a)
            i += 1
        else:
            out.append(b)
            j += 1
    if i < n_left:
        out.extend(left[i:])
    if j < n_right:
        out.extend(right[j:])
    return out


def merge_not(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Two-pointer difference: ids in ``left`` that are not in ``right``."""
    i = j = 0
    out: list[int] = []
    n_left, n_right = len(left), len(right)
    while i < n_left and j < n_right:
        a, b = left[i], right[j]
        if a == b:
            i += 1
            j += 1
        elif a < b:
            out.append(a)
            i += 1
        else:
            j += 1
    if i < n_left:
        out.extend(left[i:])
    return out


def set_and(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return sorted(set(left) & set(right))


def set_or(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return sorted(set(left) | set(right))


def set_not(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return sorted(set(left) - set(right))


def _ops(engine: str) -> tuple:
    if engine == "set":
        return set_and, set_or, set_not
    if engine == "merge":
        return merge_and, merge_or, merge_not
    raise ValueError(f"unknown engine {engine!r}")


def search(index: Index, query: str, *, engine: str = "merge") -> list[int]:
    """Evaluate ``query`` left-to-right; return matching ``doc_id``s."""
    tokens = list(tokenize(query))
    if not tokens:
        return []

    and_op, or_op, not_op = _ops(engine)
    universe = list(range(index.n_docs()))
    result: list[int] | None = None
    pending = "and"
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in OPERATORS:
            pending = tok
            i += 1
            continue
        docs = index.doc_ids_for(tok)
        if result is None:
            result = not_op(universe, docs) if pending == "not" else docs
        elif pending == "or":
            result = or_op(result, docs)
        elif pending == "not":
            result = not_op(result, docs)
        else:
            result = and_op(result, docs)
        pending = "and"
        i += 1

    if result is None:
        return []
    return list(result)


def format_hit(index: Index, doc_id: int) -> str:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        return f"{doc_id}\t?"
    return f"{doc_id}\t{meta.title}"


def _term_dfs(index: Index) -> list[tuple[str, int]]:
    # Використовуємо наш новий метод df_term заміність df
    rows = [(term, index.df_term(term)) for term in index.postings]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def benchmark_engines(index: Index, *, repeat: int = 30) -> list[str]:
    """Time merge vs set on two common terms and two rare terms."""
    ranked = _term_dfs(index)
    if len(ranked) < 4:
        return ["not enough vocabulary to benchmark"]

    common = (ranked[0][0], ranked[1][0])
    rare = (ranked[-1][0], ranked[-2][0])
    lines = [
        f"common terms (highest df): {common[0]!r} df={ranked[0][1]}, "
        f"{common[1]!r} df={ranked[1][1]}",
        f"rare terms (lowest df):    {rare[0]!r} df={ranked[-1][1]}, "
        f"{ranked[-2][1]!r} df={ranked[-2][1]}",
        "",
        f"{'pair':<16} {'query':<28} {'engine':<7} {'hits':>6} {'ms':>10}",
    ]

    pairs = [
        ("common AND", f"{common[0]} {common[1]}"),
        ("common OR", f"{common[0]} OR {common[1]}"),
        ("rare AND", f"{rare[0]} {rare[1]}"),
        ("rare OR", f"{rare[0]} OR {rare[1]}"),
    ]
    for label, query in pairs:
        for engine in ("merge", "set"):
            hits = search(index, query, engine=engine)
            t0 = time.perf_counter()
            for _ in range(repeat):
                search(index, query, engine=engine)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / repeat
            lines.append(
                f"{label:<16} {query:<28} {engine:<7} {len(hits):6d} {elapsed_ms:10.4f}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Boolean search over a saved index.")
    parser.add_argument(
        "index",
        type=Path,
        help="path written by python -m findex.index",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help='query string, e.g. "elizabeth darcy" or "holmes OR watson"',
    )
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="two-pointer merge (default) or Python set algebra",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        help="max hits to print",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="benchmark merge vs set on common and rare terms",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=30,
        help="repetitions per --bench cell",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load(args.index)
    load_elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()

    print(f"loaded:         {args.index}")
    print(f"documents:      {index.n_docs()}")
    print(f"vocabulary:     {len(index.postings)}")
    print(f"load elapsed:   {load_elapsed:.3f} s")
    print(f"peak memory:    {format_bytes(peak)} ({peak} bytes)")

    if args.bench:
        print()
        for line in benchmark_engines(index, repeat=args.repeat):
            print(line)

    if args.query is None:
        if not args.bench:
            parser.error("query is required unless --bench is set")
        tracemalloc.stop()
        return 0

    t1 = time.perf_counter()
    hits = search(index, args.query, engine=args.engine)
    search_elapsed = time.perf_counter() - t1
    tracemalloc.stop()

    print(f"engine:         {args.engine}")
    print(f"query:          {args.query}")
    print(f"hits:           {len(hits)}")
    print(f"search elapsed: {search_elapsed:.6f} s")
    print()
    for doc_id in hits[: args.limit]:
        print(format_hit(index, doc_id))
    if len(hits) > args.limit:
        print(f"... and {len(hits) - args.limit} more hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())

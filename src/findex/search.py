from __future__ import annotations

import argparse
import logging
import sys
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path

from findex.index import Index
from findex.rank import SearchResult, get_scorer, ranked_search
from findex.stats import format_bytes
from findex.store import open_index


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


def boolean_search(index: Index, query: str, *, engine: str = "merge") -> list[int]:
    """Evaluate a Boolean query; return matching ``doc_id``s (Lab 2)."""
    if not query.strip():
        return []
    from findex.query import parse

    return parse(query).evaluate(index, engine=engine)


def search(
    index: Index,
    query: str,
    *,
    engine: str = "merge",
    scorer=None,
    k: int = 10,
    snippets: bool = True,
) -> list[int] | list[SearchResult]:
    """Boolean ids when ``scorer`` is omitted; ranked ``SearchResult``s otherwise."""
    if scorer is not None:
        return ranked_search(index, query, scorer=scorer, k=k, snippets=snippets)
    return boolean_search(index, query, engine=engine)


def format_hit(index: Index, doc_id: int) -> str:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        return f"{doc_id}\t?"
    return f"{doc_id}\t{meta.title}"

from typing import Literal
from findex.rank import DocId

def search_engine(
    index: Index,  # додаємо об'єкт індексу як перший параметр, щоб логіка працювала
    query: str, 
    engine_type: Literal["boolean", "ranked"], 
    scorer_type: Literal["bm25", "tfidf"],
    k: int = 10
) -> list[DocId]:
    """High-level entrypoint that matches the requirements for the CLI."""
    if engine_type == "boolean":
        return boolean_search(index, query)
    
    # Для ranked дістаємо потрібний скорер і повертаємо список лише DocId
    from findex.rank import get_scorer
    scorer = get_scorer(scorer_type)
    results = ranked_search(index, query, scorer=scorer, k=k)
    return [hit.doc_id for hit in results]

def _term_dfs(index: Index) -> list[tuple[str, int]]:
    rows = [(term, index.df(term)) for term in index]
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
    parser = argparse.ArgumentParser(description="Boolean or ranked search over a saved index.")
    parser.add_argument(
        "index",
        type=Path,
        help="path written by python -m findex.index",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help='query string, e.g. python AND (async OR await) NOT java "event loop"',
    )
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="two-pointer merge (default) or Python set algebra (Boolean mode)",
    )
    parser.add_argument(
        "--boolean",
        action="store_true",
        help="return an unranked id list (Lab 2) instead of TF-IDF/BM25",
    )
    parser.add_argument(
        "--scorer",
        choices=("bm25", "tfidf"),
        default="bm25",
        help="ranking function (ignored with --boolean)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="top-k results (ranked) or max hits to print (Boolean)",
    )
    parser.add_argument(
        "--no-snippets",
        action="store_true",
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show @timed and lru_cache hit/miss lines",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    with open_index(args.index) as index:
        load_elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()

        print(f"loaded:         {args.index}")
        print(f"index:          {index!r}")
        print(f"documents:      {index.num_docs}")
        print(f"vocabulary:     {len(index)}")
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
        if args.boolean:
            hits = boolean_search(index, args.query, engine=args.engine)
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

        scorer = get_scorer(args.scorer)
        results = ranked_search(
            index,
            args.query,
            scorer=scorer,
            k=args.limit,
            snippets=not args.no_snippets,
        )
        search_elapsed = time.perf_counter() - t1
        tracemalloc.stop()
        print(f"scorer:         {scorer!r}")
        print(f"query:          {args.query}")
        print(f"hits:           {len(results)}")
        print(f"search elapsed: {search_elapsed:.6f} s")
        print()
        for row in results:
            print(row)
        return 0


if __name__ == "__main__":
    sys.exit(main())

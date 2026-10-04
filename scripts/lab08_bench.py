"""Lab 3 loop versus the NumPy scorer on the full Gutenberg index.

Snippets are off. One warmup call per cell, then the median of nine.
The NumPy path caches the int32 copy of a slots posting list on the first
call, so the warmup is where that copy happens and the median is the ufunc.
"""

from __future__ import annotations

import time
from pathlib import Path

from findex.query import match_doc_ids
from findex.rank import BM25, ranked_search
from findex.store import load

QUERIES = (
    ("common term", "the"),
    ("rare term", "anniversary"),
    ("3-term query", "elizabeth darcy marriage"),
)


def _median_ms(fn, repeats: int = 9) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    index = load(Path("data/index.pkl"))
    scorer = BM25()
    print(f"docs={index.num_docs} terms={len(index)} representation={index.representation}")
    print(f"{'query':<16} {'matched':>8} {'lab3_ms':>10} {'numpy_ms':>10} {'speedup':>8}")
    for label, query in QUERIES:
        matched = len(match_doc_ids(index, query))

        def lab3(query: str = query) -> None:
            ranked_search(
                index, query, scorer=scorer, k=10, snippets=False, engine="python"
            )

        def numpy_scorer(query: str = query) -> None:
            ranked_search(
                index, query, scorer=scorer, k=10, snippets=False, engine="numpy"
            )

        py_ms = _median_ms(lab3)
        np_ms = _median_ms(numpy_scorer)
        speedup = py_ms / np_ms if np_ms else float("inf")
        print(
            f"{label:<16} {matched:8d} {py_ms:10.3f} {np_ms:10.3f} {speedup:7.2f}x"
        )


if __name__ == "__main__":
    main()

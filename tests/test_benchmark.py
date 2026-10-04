"""pytest-benchmark for search and build_index on a small fixture corpus.

Marked slow so the ordinary CI test step can skip them. The benchmark job
runs this file on its own and fails if either benchmark errors.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.rank import ranked_search

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bench_corpus(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("bench-corpus")
    for i in range(40):
        words = " ".join(["alpha"] * ((i % 7) + 1) + ["beta", "gamma"])
        (root / f"{i:02d}.txt").write_text(words + "\n", encoding="utf-8")
    return root


def test_build_index_benchmark(benchmark: Callable[..., object], bench_corpus) -> None:
    def build():
        return build_index(iter_documents(bench_corpus))

    index = benchmark(build)
    assert index.num_docs == 40


def test_search_benchmark(benchmark: Callable[..., object], bench_corpus) -> None:
    index = build_index(iter_documents(bench_corpus))

    def search():
        return ranked_search(index, "alpha beta", k=5, snippets=False)

    hits = benchmark(search)
    assert hits
    assert hits[0].doc_id >= 0

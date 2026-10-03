"""TF-IDF / BM25 sanity checks, snippets, and cache hits."""

from __future__ import annotations

import logging
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.query import cache_info, match_doc_ids
from findex.rank import BM25, SearchResult, TfIdf, make_snippet, ranked_search


def _write(tmp: Path, name: str, text: str) -> None:
    (tmp / name).write_text(text, encoding="utf-8")


def test_rare_term_outranks_common_only(tmp_path: Path) -> None:
    _write(tmp_path, "rare.txt", "xyzzy appears once here")
    _write(tmp_path, "common.txt", "the the the the the filler words about nothing")
    index = build_index(iter_documents(tmp_path))
    results = ranked_search(index, "xyzzy the", scorer=TfIdf(), k=2, snippets=False)
    assert results[0].title.startswith("xyzzy")
    assert results[0].score > results[1].score


def test_bm25_twentieth_repetition_adds_almost_nothing(tmp_path: Path) -> None:
    pad = " ".join(["lorem"] * 30)
    _write(tmp_path, "tf01.txt", "needle " + pad)
    _write(tmp_path, "tf19.txt", " ".join(["needle"] * 19) + " " + pad)
    _write(tmp_path, "tf20.txt", " ".join(["needle"] * 20) + " " + pad)
    index = build_index(iter_documents(tmp_path))
    scorer = BM25(k1=1.5, b=0.75)
    ranked = {
        Path(index.doc_meta[r.doc_id].path).stem: r.score
        for r in ranked_search(index, "needle", scorer=scorer, k=3, snippets=False)
    }
    gain_early = ranked["tf19"] - ranked["tf01"]
    gain_late = ranked["tf20"] - ranked["tf19"]
    assert gain_late < 0.15 * gain_early


def test_short_doc_outranks_long_one_hit(tmp_path: Path) -> None:
    _write(tmp_path, "short.txt", "walrus")
    _write(tmp_path, "long.txt", "walrus " + " ".join(["padding"] * 400))
    index = build_index(iter_documents(tmp_path))
    tfidf = ranked_search(index, "walrus", scorer=TfIdf(), k=2, snippets=False)
    bm25 = ranked_search(index, "walrus", scorer=BM25(), k=2, snippets=False)
    bm25_order = [Path(index.doc_meta[r.doc_id].path).stem for r in bm25]
    assert bm25_order[0] == "short"
    assert bm25[0].score > bm25[1].score
    assert abs(tfidf[0].score - tfidf[1].score) < 1e-9


def test_search_result_order_and_snippet(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "the event loop runs callbacks and the event loop waits")
    index = build_index(iter_documents(tmp_path), positions=True)
    results = ranked_search(index, '"event loop"', scorer=BM25(), k=1)
    assert "**event**" in results[0].snippet
    assert "**loop**" in results[0].snippet
    a = SearchResult(doc_id=1, score=1.0, title="a")
    b = SearchResult(doc_id=2, score=3.0, title="b")
    assert sorted([b, a]) == [a, b]


def test_scorers_are_interchangeable(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "python async await")
    index = build_index(iter_documents(tmp_path))
    for scorer in (TfIdf(), BM25()):
        hits = ranked_search(index, "python", scorer=scorer, k=5, snippets=False)
        assert hits and hits[0].doc_id == 0


def test_lru_cache_hit_logged(tmp_path: Path, caplog) -> None:
    _write(tmp_path, "a.txt", "python async")
    index = build_index(iter_documents(tmp_path))
    caplog.set_level(logging.INFO)
    match_doc_ids(index, "python")
    match_doc_ids(index, "python")
    assert "lru_cache hit" in caplog.text
    info = cache_info()
    assert info.hits >= 1

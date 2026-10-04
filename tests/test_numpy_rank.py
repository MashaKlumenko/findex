"""NumPy BM25/TF-IDF matches the Lab 3 loop, including on buffer postings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from findex.corpus import iter_documents
from findex.index import build_index
from findex.rank import BM25, TfIdf, ranked_search
from findex.store import load, save


def _corpus(tmp: Path) -> None:
    (tmp / "short.txt").write_text("walrus\n", encoding="utf-8")
    (tmp / "long.txt").write_text(
        "walrus " + " ".join(["padding"] * 40) + "\n", encoding="utf-8"
    )
    (tmp / "rare.txt").write_text(
        "xyzzy appears once beside the walrus\n", encoding="utf-8"
    )
    (tmp / "common.txt").write_text(
        "the the the the filler words about nothing else\n", encoding="utf-8"
    )
    (tmp / "phrase.txt").write_text(
        "the event loop runs the event loop twice\n", encoding="utf-8"
    )


def _assert_same(index, query: str, scorer) -> None:
    python = ranked_search(
        index, query, scorer=scorer, k=10, snippets=False, engine="python"
    )
    numpy = ranked_search(
        index, query, scorer=scorer, k=10, snippets=False, engine="numpy"
    )
    assert [hit.doc_id for hit in python] == [hit.doc_id for hit in numpy]
    for left, right in zip(python, numpy, strict=True):
        assert left.score == pytest_approx(right.score)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-9, abs=1e-12)


def test_numpy_order_matches_lab3(tmp_path: Path) -> None:
    _corpus(tmp_path)
    index = build_index(iter_documents(tmp_path), positions=True)
    queries = (
        "walrus",
        "the",
        "xyzzy the",
        "walrus OR xyzzy",
        "the NOT walrus",
        '"event loop"',
    )
    for query in queries:
        _assert_same(index, query, BM25())
        _assert_same(index, query, TfIdf())


def test_numpy_postings_are_int32_and_roundtrip(tmp_path: Path) -> None:
    _corpus(tmp_path)
    index = build_index(iter_documents(tmp_path), representation="numpy")
    ids, tfs = index.posting_arrays("walrus")
    assert ids is not None and tfs is not None
    assert ids.dtype == np.int32
    assert tfs.dtype == np.int32
    assert index.doc_len_array.dtype == np.int32
    assert index.doc_len_array.shape == (index.num_docs,)

    path = tmp_path / "index.json"
    save(index, path)
    loaded = load(path)
    assert loaded.representation == "numpy"
    assert loaded.doc_ids_for("walrus") == index.doc_ids_for("walrus")
    _assert_same(loaded, "walrus the", BM25())


def test_array_postings_are_a_view(tmp_path: Path) -> None:
    _corpus(tmp_path)
    index = build_index(iter_documents(tmp_path), representation="array")
    ids, _tfs = index.posting_arrays("the")
    assert ids is not None
    assert ids.dtype == np.int32
    assert not ids.flags.owndata

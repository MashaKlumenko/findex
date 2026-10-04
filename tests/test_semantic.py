"""Chunking, best-chunk cosine, and reciprocal rank fusion. No model download."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from findex.corpus import iter_documents
from findex.index import build_index
from findex.semantic import (
    Embeddings,
    chunk_text,
    embed_index,
    hybrid_search,
    l2_normalize,
    reciprocal_rank_fusion,
    semantic_search,
)


def test_chunk_text_windows() -> None:
    short = "one two three"
    assert chunk_text(short, size=10, overlap=2) == ["one two three"]
    tokens = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_text(tokens, size=4, overlap=1)
    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1].split()[0] == "w3"
    assert chunks[-1].split()[-1] == "w9"


def test_best_chunk_wins(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("cats sleep softly\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("markets fell sharply\n", encoding="utf-8")
    index = build_index(iter_documents(tmp_path))
    vectors = l2_normalize(
        np.array(
            [
                [1.0, 0.0],
                [0.2, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    # Document 0 owns the first two chunks. Its best chunk is the unit x axis.
    embeddings = Embeddings(vectors, np.array([0, 0, 1], dtype=np.int32), model="test")

    def encode(texts: list[str]) -> np.ndarray:
        assert len(texts) == 1
        return np.array([[1.0, 0.0]], dtype=np.float32)

    hits = semantic_search(index, embeddings, "felines", encode, k=2, snippets=False)
    assert [hit.doc_id for hit in hits] == [0, 1]
    assert hits[0].score == pytest_approx(1.0)


def test_rrf_prefers_the_document_both_lists_rank_high() -> None:
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]], k=60)
    assert [doc_id for doc_id, _score in fused] == [1, 0, 2]


def test_hybrid_runs_on_a_fake_encoder(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha alpha beta\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta gamma gamma\n", encoding="utf-8")
    index = build_index(iter_documents(tmp_path))

    def encode(texts):
        rows = []
        for text in texts:
            vec = np.zeros(4, dtype=np.float32)
            for token in str(text).split():
                vec[sum(ord(ch) for ch in token) % 4] += 1.0
            rows.append(vec)
        return np.vstack(rows)

    embeddings = embed_index(index, encode, model="hash")
    hits = hybrid_search(
        index, embeddings, "alpha", encode, k=2, snippets=False, depth=5
    )
    assert hits
    assert hits[0].doc_id == 0
    path = tmp_path / "emb"
    embeddings.save(path)
    loaded = Embeddings.load(path)
    assert loaded.vectors.shape == embeddings.vectors.shape
    assert loaded.model == "hash"


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-6, abs=1e-6)

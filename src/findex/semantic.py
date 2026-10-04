"""Chunk embeddings, cosine search, and reciprocal rank fusion.

Vectors are L2-normalized once. Cosine similarity is then a dot product,
and one query is the matrix-vector product ``E @ q``. A document's score
is its best chunk.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from findex.index import Index
from findex.rank import BM25, Scorer, SearchResult, _pack_results, ranked_search
from findex.tokenize import tokenize

Encode = Callable[[Sequence[str]], np.ndarray]
DEFAULT_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60


class EmbeddingsUnavailable(RuntimeError):
    """Semantic mode was asked for and the matrix or the model is missing."""


@dataclass(frozen=True)
class Embeddings:
    """Normalized chunk matrix plus the document that owns each row."""

    vectors: np.ndarray
    doc_ids: np.ndarray
    model: str

    def save(self, directory: Path | str) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "vectors.npy", self.vectors)
        np.save(root / "doc_ids.npy", self.doc_ids)
        dim = int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0
        (root / "meta.json").write_text(
            json.dumps({"model": self.model, "dim": dim}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path | str) -> Embeddings:
        root = Path(directory)
        vectors = np.load(root / "vectors.npy")
        doc_ids = np.load(root / "doc_ids.npy")
        meta_path = root / "meta.json"
        model = DEFAULT_MODEL
        if meta_path.is_file():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("model"), str):
                model = payload["model"]
        return cls(
            vectors=np.asarray(vectors, dtype=np.float32),
            doc_ids=np.asarray(doc_ids, dtype=np.int32),
            model=model,
        )


def chunk_text(text: str, *, size: int = 256, overlap: int = 32) -> list[str]:
    """Split ``text`` into token windows. A short passage stays one chunk."""
    if size < 1:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    tokens = list(tokenize(text))
    if not tokens:
        return []
    if len(tokens) <= size:
        return [" ".join(tokens)]
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit rows. A zero row stays zero."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, np.float32(1e-12))
    return (arr / norms).astype(np.float32, copy=False)


def embed_index(
    index: Index,
    encode: Encode,
    *,
    model: str = DEFAULT_MODEL,
    chunk_size: int = 256,
    overlap: int = 32,
) -> Embeddings:
    """Chunk every document, encode, L2-normalize, and keep the doc id of each row."""
    texts: list[str] = []
    owners: list[int] = []
    for doc_id in sorted(index.doc_meta):
        for chunk in chunk_text(
            index.document_text(doc_id), size=chunk_size, overlap=overlap
        ):
            texts.append(chunk)
            owners.append(doc_id)
    if not texts:
        empty = np.zeros((0, 1), dtype=np.float32)
        return Embeddings(empty, np.zeros(0, dtype=np.int32), model)
    vectors = l2_normalize(encode(texts))
    if vectors.shape[0] != len(owners):
        raise ValueError(
            f"encoder returned {vectors.shape[0]} rows for {len(owners)} chunks"
        )
    return Embeddings(vectors, np.asarray(owners, dtype=np.int32), model)


def encode_query(encode: Encode, text: str) -> np.ndarray:
    raw = np.asarray(encode([text]), dtype=np.float32)
    return l2_normalize(raw).reshape(-1)


def best_chunk_scores(
    embeddings: Embeddings, query: np.ndarray, n_docs: int
) -> np.ndarray:
    """Cosine of each document's nearest chunk. Missing documents stay ``-inf``."""
    best = np.full(n_docs, -np.inf, dtype=np.float64)
    if embeddings.vectors.size == 0 or embeddings.doc_ids.size == 0:
        return best
    sims = embeddings.vectors @ query.astype(np.float32, copy=False)
    np.maximum.at(best, embeddings.doc_ids, sims.astype(np.float64, copy=False))
    return best


def semantic_ranking(
    embeddings: Embeddings,
    query_vector: np.ndarray,
    *,
    n_docs: int,
    k: int,
) -> list[tuple[int, float]]:
    scores = best_chunk_scores(embeddings, query_vector, n_docs)
    present = np.zeros(n_docs, dtype=np.bool_)
    if embeddings.doc_ids.size:
        present[embeddings.doc_ids] = True
    from findex.numpy_rank import argpartition_top

    return argpartition_top(scores, np.flatnonzero(present), k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse ranked id lists. Score of a document is ``Σ 1 / (k + rank)``."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, score) for doc_id, score in ordered]


def semantic_search(
    index: Index,
    embeddings: Embeddings,
    query: str,
    encode: Encode,
    *,
    k: int = 10,
    snippets: bool = True,
) -> list[SearchResult]:
    if not query.strip() or k <= 0:
        return []
    vector = encode_query(encode, query)
    n_docs = int(index.doc_len_array.shape[0]) if index.doc_meta else 0
    ranked = semantic_ranking(embeddings, vector, n_docs=n_docs, k=k)
    terms = list(tokenize(query))
    return _pack_results(index, ranked, terms, snippets=snippets)


def hybrid_search(
    index: Index,
    embeddings: Embeddings,
    query: str,
    encode: Encode,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    snippets: bool = True,
    depth: int = 200,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    """BM25 (or TF-IDF) fused with the semantic ranking by RRF."""
    if not query.strip() or k <= 0:
        return []
    if scorer is None:
        scorer = BM25()
    pool = max(k, depth)
    keyword = ranked_search(
        index, query, scorer=scorer, k=pool, snippets=False, engine="numpy"
    )
    vector = encode_query(encode, query)
    n_docs = int(index.doc_len_array.shape[0]) if index.doc_meta else 0
    semantic = semantic_ranking(embeddings, vector, n_docs=n_docs, k=pool)
    fused = reciprocal_rank_fusion(
        ([hit.doc_id for hit in keyword], [doc_id for doc_id, _score in semantic]),
        k=rrf_k,
    )
    terms = list(tokenize(query))
    return _pack_results(index, fused[:k], terms, snippets=snippets)


def load_encoder(model_name: str = DEFAULT_MODEL, *, progress: bool = False) -> Encode:
    """Load all-MiniLM. ONNX via fastembed, then sentence-transformers if that imports.

    The PyTorch wheel's ``shm.dll`` does not load on this Windows machine, so
    the ONNX build of the same checkpoint is the one that actually runs.
    """
    errors: list[str] = []
    for factory in (_fastembed_encoder, _sentence_encoder):
        try:
            return factory(model_name, progress=progress)
        except (ImportError, OSError) as exc:
            errors.append(f"{factory.__name__}: {exc}")
    raise EmbeddingsUnavailable(
        "semantic search needs fastembed or sentence-transformers. "
        + " | ".join(errors)
    )


def _fastembed_encoder(model_name: str, *, progress: bool) -> Encode:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=_onnx_model_name(model_name))

    def encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if not progress:
            rows = list(model.embed(list(texts), batch_size=64))
            return np.asarray(rows, dtype=np.float32)
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

        batches: list[np.ndarray] = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as bar:
            task = bar.add_task("embedding chunks", total=len(texts))
            for start in range(0, len(texts), 64):
                batch = list(texts[start : start + 64])
                batches.append(np.asarray(list(model.embed(batch)), dtype=np.float32))
                bar.advance(task, len(batch))
        return np.vstack(batches)

    return encode


def _sentence_encoder(model_name: str, *, progress: bool) -> Encode:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_sentence_model_name(model_name))

    def encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = int(model.get_sentence_embedding_dimension() or 1)
            return np.zeros((0, dim), dtype=np.float32)
        rows = model.encode(
            list(texts),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=progress,
        )
        return np.asarray(rows, dtype=np.float32)

    return encode


def _onnx_model_name(model_name: str) -> str:
    short = model_name.rsplit("/", 1)[-1]
    if short == DEFAULT_MODEL:
        return "sentence-transformers/all-MiniLM-L6-v2"
    return model_name


def _sentence_model_name(model_name: str) -> str:
    short = model_name.rsplit("/", 1)[-1]
    if short == DEFAULT_MODEL and "/" not in model_name:
        return DEFAULT_MODEL
    return model_name


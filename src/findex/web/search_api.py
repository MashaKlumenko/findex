"""Pagination over keyword, semantic, and hybrid search."""

from __future__ import annotations

import time
from typing import Literal

from findex.index import Index
from findex.query import QuerySyntaxError, match_doc_ids
from findex.rank import SearchResult as RankedHit
from findex.rank import get_scorer, make_snippet, ranked_search
from findex.semantic import (
    Embeddings,
    EmbeddingsUnavailable,
    Encode,
    hybrid_search,
    semantic_search,
)
from findex.tokenize import tokenize
from findex.web.schemas import SearchResponse, SearchResult

ScorerName = Literal["bm25", "tfidf"]
SearchMode = Literal["keyword", "semantic", "hybrid"]


def run_search(
    index: Index,
    query: str,
    *,
    k: int,
    scorer: ScorerName,
    page: int,
    mode: SearchMode = "keyword",
    embeddings: Embeddings | None = None,
    encode: Encode | None = None,
) -> SearchResponse:
    """Score the query once, then slice one page out of the ranked list.

    Keyword search still uses the memoized Boolean match for ``total``.
    Semantic search ranks every embedded document. Hybrid fuses the top of
    each list with reciprocal rank fusion and reports how many documents
    entered that fusion.
    """
    started = time.perf_counter()
    need = page * k
    if mode == "keyword":
        total = len(match_doc_ids(index, query))
        ranked = ranked_search(
            index,
            query,
            scorer=get_scorer(scorer),
            k=need,
            snippets=True,
        )
    elif mode == "semantic":
        matrix, encoder = _ready(embeddings, encode)
        total = len({int(doc_id) for doc_id in matrix.doc_ids})
        ranked = semantic_search(
            index, matrix, query, encoder, k=need, snippets=True
        )
    else:
        matrix, encoder = _ready(embeddings, encode)
        ranked = hybrid_search(
            index,
            matrix,
            query,
            encoder,
            scorer=get_scorer(scorer),
            k=max(need, 200),
            snippets=False,
        )
        total = len(ranked)

    offset = (page - 1) * k
    page_hits = ranked[offset : offset + k]
    if mode == "hybrid":
        terms = list(tokenize(query))
        for hit in page_hits:
            hit.snippet = make_snippet(index.document_text(hit.doc_id), terms)
    took_ms = (time.perf_counter() - started) * 1000
    return SearchResponse(
        query=query,
        total=total,
        page=page,
        k=k,
        took_ms=round(took_ms, 3),
        mode=mode,
        results=[_hit(hit) for hit in page_hits],
    )


def _ready(
    embeddings: Embeddings | None, encode: Encode | None
) -> tuple[Embeddings, Encode]:
    if embeddings is None or encode is None:
        raise EmbeddingsUnavailable(
            "Semantic search needs an embeddings directory and a model. "
            "Run `findex embed` and set EMBEDDINGS_PATH."
        )
    return embeddings, encode


def _hit(hit: RankedHit) -> SearchResult:
    return SearchResult(
        doc_id=hit.doc_id,
        title=hit.title,
        score=hit.score,
        snippet=hit.snippet,
    )


__all__ = ["EmbeddingsUnavailable", "QuerySyntaxError", "run_search"]

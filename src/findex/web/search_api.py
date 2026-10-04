"""Pagination over the Lab 3 ranker. Dataclasses stay on this side of the seam."""

from __future__ import annotations

import time
from typing import Literal

from findex.index import Index
from findex.query import QuerySyntaxError, match_doc_ids
from findex.rank import SearchResult as RankedHit
from findex.rank import get_scorer, ranked_search
from findex.web.schemas import SearchResponse, SearchResult

ScorerName = Literal["bm25", "tfidf"]


def run_search(
    index: Index,
    query: str,
    *,
    k: int,
    scorer: ScorerName,
    page: int,
) -> SearchResponse:
    """Score the query once, then slice one page out of the ranked list.

    ``match_doc_ids`` is memoized, so the total and the ranker share one
    parse. A syntax error propagates as ``QuerySyntaxError``; the route
    turns that into HTTP 400.
    """
    started = time.perf_counter()
    total = len(match_doc_ids(index, query))
    ranked: list[RankedHit] = ranked_search(
        index,
        query,
        scorer=get_scorer(scorer),
        k=page * k,
        snippets=True,
    )
    offset = (page - 1) * k
    page_hits = ranked[offset : offset + k]
    took_ms = (time.perf_counter() - started) * 1000
    return SearchResponse(
        query=query,
        total=total,
        page=page,
        k=k,
        took_ms=round(took_ms, 3),
        results=[
            SearchResult(
                doc_id=hit.doc_id,
                title=hit.title,
                score=hit.score,
                snippet=hit.snippet,
            )
            for hit in page_hits
        ],
    )


__all__ = ["QuerySyntaxError", "run_search"]

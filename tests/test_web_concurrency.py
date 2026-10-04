"""20 concurrent searches: ``def`` must not stall the event loop.

The handler sleeps for 50 ms on top of a real search, standing in for the
blocking file reads inside snippet generation. ``async def`` runs that sleep
on the loop, so the requests queue. ``def`` runs it in the threadpool, so
they overlap.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal

import pytest
from httpx import ASGITransport, AsyncClient

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.web import search_api
from findex.web.app import create_app
from findex.web.deps import get_index
from findex.web.schemas import SearchResponse
from findex.web.settings import Settings

ScorerName = Literal["bm25", "tfidf"]


def test_threaded_search_overlaps_blocking_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(4):
        (tmp_path / f"{i}.txt").write_text(f"alpha passage {i}\n", encoding="utf-8")
    index = build_index(iter_documents(tmp_path), positions=False)
    real = search_api.run_search

    def slow(
        index: Index,
        query: str,
        *,
        k: int,
        scorer: ScorerName,
        page: int,
        **kwargs: object,
    ) -> SearchResponse:
        time.sleep(0.05)
        return real(index, query, k=k, scorer=scorer, page=page, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_api, "run_search", slow)

    async def elapsed(blocking_async: bool) -> float:
        application = create_app(
            Settings(index_path=None, findex_search_async=blocking_async)
        )
        application.dependency_overrides[get_index] = lambda: index
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            started = time.perf_counter()
            responses = await asyncio.gather(
                *[
                    client.get("/search", params={"q": "alpha", "k": 3})
                    for _ in range(20)
                ]
            )
            duration = time.perf_counter() - started
        assert all(response.status_code == 200 for response in responses)
        return duration

    blocking = asyncio.run(elapsed(True))
    threaded = asyncio.run(elapsed(False))
    print(f"20 concurrent /search: async def {blocking:.3f}s, def {threaded:.3f}s")
    assert blocking > 0.8
    assert threaded < blocking / 2

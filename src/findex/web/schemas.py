"""Pydantic models for the HTTP boundary.

Internal search still uses the Lab 2–3 dataclasses. These models only
validate what comes in from the query string and what goes out as JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Query-string parameters for ``GET /search``."""

    q: str = Field(min_length=1, max_length=200, description="Ranked query")
    k: int = Field(default=10, ge=1, le=100, description="Page size")
    scorer: Literal["bm25", "tfidf"] = "bm25"
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    page: int = Field(default=1, ge=1, le=1000)


class SearchResult(BaseModel):
    """One ranked hit, serialized for the client."""

    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    page: int
    k: int
    took_ms: float
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    results: list[SearchResult]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsOut(BaseModel):
    documents: int
    vocabulary: int
    avg_doc_length: float
    index_bytes: int | None
    uptime_seconds: float
    representation: str


class HealthOut(BaseModel):
    status: Literal["ok", "unavailable"]

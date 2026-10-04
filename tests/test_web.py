# TestClient's httpx stubs are untyped under the current Starlette pin.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""HTTP API tests. The index is a fixture, never a file on disk."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.web.app import create_app
from findex.web.deps import get_index
from findex.web.settings import Settings


@pytest.fixture
def tiny_index(tmp_path: Path) -> Index:
    for i in range(12):
        text = f"Passage {i}\n" + " ".join(["alpha"] * (i + 1) + ["beta"])
        if i == 3:
            text += " <script>alert(1)</script>"
        (tmp_path / f"{i:02d}.txt").write_text(text, encoding="utf-8")
    return build_index(iter_documents(tmp_path), positions=True)


@pytest.fixture
def application(tiny_index: Index) -> FastAPI:
    app = create_app(Settings(index_path=None))
    app.dependency_overrides[get_index] = lambda: tiny_index
    return app


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    with TestClient(application) as test_client:
        yield test_client


def test_search_success(client: TestClient) -> None:
    response = client.get("/search", params={"q": "alpha", "k": 5, "page": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "alpha"
    assert body["total"] == 12
    assert body["page"] == 1
    assert body["k"] == 5
    assert body["took_ms"] >= 0
    assert len(body["results"]) == 5
    hit = body["results"][0]
    assert set(hit) == {"doc_id", "title", "score", "snippet"}
    assert "**alpha**" in hit["snippet"]
    assert response.headers["x-request-id"]


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "alpha", "k": 5, "page": 1}).json()
    second = client.get("/search", params={"q": "alpha", "k": 5, "page": 2}).json()
    third = client.get("/search", params={"q": "alpha", "k": 5, "page": 3}).json()
    assert len(second["results"]) == 5
    assert len(third["results"]) == 2
    ids_first = {hit["doc_id"] for hit in first["results"]}
    ids_second = {hit["doc_id"] for hit in second["results"]}
    assert ids_first.isdisjoint(ids_second)
    assert second["total"] == 12


def test_home_offers_hybrid(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "hybrid" in response.text
    assert 'name="mode"' in response.text


def test_semantic_without_embeddings_is_400(client: TestClient) -> None:
    response = client.get("/search", params={"q": "alpha", "mode": "semantic"})
    assert response.status_code == 400
    assert "embeddings" in response.json()["detail"].lower()


def test_tfidf_scorer(client: TestClient) -> None:
    response = client.get("/search", params={"q": "beta", "scorer": "tfidf"})
    assert response.status_code == 200
    assert response.json()["results"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "alpha", "k": 0},
        {"q": "alpha", "k": 101},
        {"q": "alpha", "k": "abc"},
        {"q": "alpha", "scorer": "cosine"},
        {"q": "alpha", "page": 0},
        {"q": "x" * 201},
    ],
)
def test_search_422(client: TestClient, params: dict[str, object]) -> None:
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_bad_query_is_400(client: TestClient) -> None:
    response = client.get("/search", params={"q": "@@@"})
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Traceback" not in response.text


def test_unknown_document_is_404(client: TestClient) -> None:
    response = client.get("/docs/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "document not found"


def test_document_id_must_be_an_int(client: TestClient) -> None:
    assert client.get("/docs/nope").status_code == 422


def test_document_json(client: TestClient) -> None:
    response = client.get("/docs/0")
    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == 0
    assert "alpha" in body["text"]
    assert body["title"]


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 12
    assert body["vocabulary"] >= 3
    assert body["avg_doc_length"] > 0
    assert body["index_bytes"] is None
    assert body["uptime_seconds"] >= 0
    assert body["representation"]


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_503_when_index_is_overridden_away() -> None:
    application = create_app(Settings(index_path=None))
    application.dependency_overrides[get_index] = lambda: None
    with TestClient(application) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_search_503_when_nothing_is_loaded() -> None:
    application = create_app(Settings(index_path=None))
    with TestClient(application) as test_client:
        response = test_client.get("/search", params={"q": "alpha"})
    assert response.status_code == 503
    assert response.json()["detail"] == "index not loaded"


def test_missing_index_file_fails_at_startup(tmp_path: Path) -> None:
    application = create_app(Settings(index_path=tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        with TestClient(application):
            pass


def test_unexpected_exception_is_a_clean_500(
    application: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    @application.get("/explode")
    def explode() -> None:
        raise RuntimeError("boom-secret-stack")

    logger = logging.getLogger("findex.web")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger="findex.web")
    try:
        with TestClient(application, raise_server_exceptions=False) as test_client:
            response = test_client.get("/explode")
    finally:
        logger.removeHandler(caplog.handler)

    assert response.status_code == 500
    assert "boom-secret-stack" not in response.text
    assert "Traceback" not in response.text
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["request_id"]
    assert body["request_id"] in caplog.text
    assert "boom-secret-stack" in caplog.text


def test_openapi_and_docs_page(client: TestClient) -> None:
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "/search" in paths
    assert "/stats" in paths
    assert "/health" in paths
    assert "/docs/{doc_id}" in paths
    page = client.get("/docs")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]


def test_home_page_highlights_and_escapes(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert 'id="q"' in home.text
    assert "Search the novels" in home.text

    results = client.get("/", params={"q": "alpha", "k": 5})
    assert results.status_code == 200
    assert "<mark>" in results.text
    assert "<script>alert(1)</script>" not in results.text

    wide = client.get("/", params={"q": "alpha", "k": 1, "page": 6})
    assert "…" in wide.text

    fragment = client.get(
        "/",
        params={"q": "alpha", "k": 5},
        headers={"HX-Request": "true"},
    )
    assert fragment.text.strip().startswith("<section")
    assert "<html" not in fragment.text.lower()

    passage = client.get("/doc/3", params={"q": "alpha"})
    assert passage.status_code == 200
    assert "<script>alert(1)</script>" not in passage.text
    assert "&lt;script&gt;" in passage.text

    missing = client.get("/doc/99999")
    assert missing.status_code == 404
    assert "No passage" in missing.text


def test_request_id_round_trip(client: TestClient) -> None:
    echoed = client.get("/health", headers={"X-Request-Id": "lab07-req"})
    assert echoed.headers["x-request-id"] == "lab07-req"
    generated = client.get("/health", headers={"X-Request-Id": "bad id"})
    assert generated.headers["x-request-id"] != "bad id"
    assert " " not in generated.headers["x-request-id"]


def test_lifespan_loads_the_index_file(tiny_index: Index, tmp_path: Path) -> None:
    from findex.store import save

    path = tmp_path / "index.json"
    save(tiny_index, path)
    application = create_app(Settings(index_path=path))
    with TestClient(application) as test_client:
        response = test_client.get("/search", params={"q": "alpha", "k": 2})
        assert response.status_code == 200
        assert response.json()["total"] == 12
        stats = test_client.get("/stats").json()
        assert stats["index_bytes"] == path.stat().st_size
    assert application.state.index is None


def test_html_errors_for_syntax_and_unloaded_index(tiny_index: Index) -> None:
    unloaded = create_app(Settings(index_path=None))
    with TestClient(unloaded) as test_client:
        missing = test_client.get("/", params={"q": "alpha"})
        assert missing.status_code == 503
        assert "not loaded" in missing.text
        passage = test_client.get("/doc/1")
        assert passage.status_code == 503

    loaded = create_app(Settings(index_path=None))
    loaded.dependency_overrides[get_index] = lambda: tiny_index
    with TestClient(loaded) as test_client:
        bad = test_client.get("/", params={"q": "@@@"})
        assert bad.status_code == 400
        assert "unexpected character" in bad.text


def test_blank_index_path_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from findex.web.settings import load_settings

    monkeypatch.setenv("INDEX_PATH", "   ")
    load_settings.cache_clear()
    try:
        assert Settings().index_path is None
    finally:
        load_settings.cache_clear()


def test_serve_command_exports_index_path_and_calls_uvicorn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn
    from typer.testing import CliRunner

    from findex.cli import app

    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["index_path"] = os.environ.get("INDEX_PATH")

    monkeypatch.setattr(uvicorn, "run", fake_run)
    previous = os.environ.get("INDEX_PATH")
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    try:
        result = CliRunner().invoke(
            app,
            ["serve", "--port", "8123", "--workers", "2", "--index", str(index)],
        )
    finally:
        if previous is None:
            os.environ.pop("INDEX_PATH", None)
        else:
            os.environ["INDEX_PATH"] = previous

    assert result.exit_code == 0, result.output
    assert captured["args"] == ("findex.web:app",)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["port"] == 8123
    assert kwargs["workers"] == 2
    assert captured["index_path"] == str(index.resolve())
    assert os.environ.get("INDEX_PATH") == previous

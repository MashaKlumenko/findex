"""ASGI application: one index for the process, JSON at the edges, HTML for people."""

from __future__ import annotations

import math
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from findex.index import DocMeta, Index
from findex.query import QuerySyntaxError
from findex.semantic import Embeddings, EmbeddingsUnavailable, Encode, load_encoder
from findex.store import load
from findex.web import search_api
from findex.web.deps import get_index, get_settings, require_index
from findex.web.highlight import (
    book_name,
    display_title,
    highlight_snippet,
    highlight_text,
)
from findex.web.log import configure_logging, get_logger
from findex.web.schemas import (
    DocumentOut,
    HealthOut,
    SearchRequest,
    SearchResponse,
    StatsOut,
)
from findex.web.schemas import (
    SearchResult as ApiSearchResult,
)
from findex.web.settings import Settings, load_settings

APP_VERSION = "1.0.0"
WEB_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("elizabeth", "Elizabeth"),
    ("monster", "the monster"),
    ("dorian gray", "Dorian Gray"),
    ('"sherlock holmes"', "Sherlock Holmes"),
    ("alice", "Alice"),
    ("dracula", "Dracula"),
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
# Jinja's filter map is typed as the built-in filters only.
templates.env.filters["mark_snippet"] = highlight_snippet  # type: ignore[assignment]
templates.env.filters["mark_text"] = highlight_text  # type: ignore[assignment]
templates.env.globals["app_version"] = APP_VERSION  # type: ignore[index]


class RequestIdMiddleware:
    """Give every request an id, log the outcome, and never leak a traceback."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status_code = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            path = scope.get("path", "")
            logger = get_logger()
            log = logger.debug if path == "/health" else logger.info
            log(
                "%s %s %s %.1fms request_id=%s",
                scope.get("method", "-"),
                path,
                status_code,
                elapsed_ms,
                request_id,
            )


def _incoming_request_id(scope: Scope) -> str:
    raw = scope.get("headers", [])
    rows = cast(list[tuple[object, object]], raw) if isinstance(raw, list) else []
    for key, value in rows:
        if key != b"x-request-id" or not isinstance(value, bytes):
            continue
        text = value.decode("latin-1", errors="ignore").strip()
        if _REQUEST_ID.fullmatch(text):
            return text
        break
    return uuid.uuid4().hex[:16]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass ``settings`` in tests so the process env is ignored."""
    resolved = settings if settings is not None else load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load the index once. Uvicorn runs this around the whole process."""
        configure_logging(resolved.log_level)
        logger = get_logger()
        app.state.started_at = time.monotonic()
        index: Index | None = None
        path = resolved.index_path
        if path is None:
            logger.warning("INDEX_PATH is unset; /health will return 503")
        else:
            logger.info("loading index path=%s", path)
            index = load(path)
            app.state.index = index
            logger.info(
                "index ready documents=%s vocabulary=%s",
                index.num_docs,
                len(index),
            )
        emb_path = resolved.embeddings_path
        if emb_path is not None:
            try:
                app.state.embeddings = Embeddings.load(emb_path)
                logger.info(
                    "embeddings ready path=%s model=%s",
                    emb_path,
                    app.state.embeddings.model,
                )
            except Exception:
                logger.exception("embeddings not loaded path=%s", emb_path)
        try:
            yield
        finally:
            app.state.index = None
            if index is not None:
                index.close()
                logger.info("index closed")

    application = FastAPI(
        title="findex",
        version=APP_VERSION,
        summary="Search an inverted index over HTTP.",
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.index = None
    application.state.embeddings = None
    application.state.encode = None
    application.state.started_at = time.monotonic()

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestIdMiddleware)
    application.add_exception_handler(Exception, _unhandled)

    _register_search(application, blocking_async=resolved.findex_search_async)
    _register_api(application)
    _register_pages(application)

    if STATIC_DIR.is_dir():
        application.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="static",
        )
    return application


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Log the stack with the request id. The client gets neither."""
    request_id = getattr(request.state, "request_id", None) or "-"
    get_logger().exception(
        "unhandled exception request_id=%s",
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


def _register_search(application: FastAPI, *, blocking_async: bool) -> None:
    """``def`` is the production route: FastAPI runs it in a threadpool.

    Ranked search is CPU work plus blocking reads of the passage text.
    An ``async def`` that calls it directly holds the event loop, so the
    other 19 of 20 concurrent requests wait in line. ``FINDEX_SEARCH_ASYNC``
    turns that variant on for the load-test row and for nothing else.
    """

    if blocking_async:

        @application.get("/search", response_model=SearchResponse, tags=["search"])
        async def search_blocking(
            params: Annotated[SearchRequest, Query()],
            index: Annotated[Index, Depends(require_index)],
            request: Request,
        ) -> SearchResponse:
            return _perform_search(params, index, request)

    else:

        @application.get("/search", response_model=SearchResponse, tags=["search"])
        def search(
            params: Annotated[SearchRequest, Query()],
            index: Annotated[Index, Depends(require_index)],
            request: Request,
        ) -> SearchResponse:
            return _perform_search(params, index, request)


def _perform_search(
    params: SearchRequest, index: Index, request: Request
) -> SearchResponse:
    try:
        return search_api.run_search(
            index,
            params.q,
            k=params.k,
            scorer=params.scorer,
            page=params.page,
            mode=params.mode,
            embeddings=getattr(request.app.state, "embeddings", None),
            encode=_encoder(request) if params.mode != "keyword" else None,
        )
    except QuerySyntaxError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmbeddingsUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _encoder(request: Request) -> Encode | None:
    """Load the embedding model on the first semantic request and keep it."""
    cached = getattr(request.app.state, "encode", None)
    if cached is not None:
        return cached
    embeddings = getattr(request.app.state, "embeddings", None)
    if not isinstance(embeddings, Embeddings):
        return None
    encode = load_encoder(embeddings.model)
    request.app.state.encode = encode
    return encode


def _register_api(application: FastAPI) -> None:
    @application.get(
        "/docs/{doc_id}",
        response_model=DocumentOut,
        tags=["documents"],
        responses={404: {"description": "Unknown document"}},
    )
    def read_document(
        doc_id: int,
        index: Annotated[Index, Depends(require_index)],
    ) -> DocumentOut:
        found = _lookup(index, doc_id)
        if found is None:
            raise HTTPException(status_code=404, detail="document not found")
        meta, text = found
        return DocumentOut(
            doc_id=doc_id,
            title=meta.title,
            path=meta.path,
            text=text,
        )

    @application.get("/stats", response_model=StatsOut, tags=["meta"])
    def stats(
        request: Request,
        index: Annotated[Index, Depends(require_index)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> StatsOut:
        return StatsOut(
            documents=index.num_docs,
            vocabulary=len(index),
            avg_doc_length=round(index.avg_doc_length, 2),
            index_bytes=_index_bytes(settings),
            uptime_seconds=_uptime(request),
            representation=index.representation,
        )

    @application.get(
        "/health",
        response_model=HealthOut,
        tags=["meta"],
        responses={503: {"model": HealthOut}},
    )
    def health(index: Annotated[Index | None, Depends(get_index)]) -> JSONResponse:
        if index is None:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})


def _register_pages(application: FastAPI) -> None:
    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        index: Annotated[Index | None, Depends(get_index)],
        q: Annotated[str, Query(max_length=200)] = "",
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        return _render_search(
            request,
            q=q,
            k=k,
            scorer=scorer,
            mode=mode,
            page=page,
            index=index,
        )

    @application.get(
        "/doc/{doc_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def document_page(
        doc_id: int,
        request: Request,
        index: Annotated[Index | None, Depends(get_index)],
        q: Annotated[str, Query(max_length=200)] = "",
    ) -> HTMLResponse:
        if index is None:
            return _html(
                request,
                "document.html",
                {"doc_id": doc_id, "missing": True, "unloaded": True, "q": q},
                status_code=503,
            )
        found = _lookup(index, doc_id)
        if found is None:
            return _html(
                request,
                "document.html",
                {"doc_id": doc_id, "missing": True, "unloaded": False, "q": q},
                status_code=404,
            )
        meta, text = found
        book = book_name(meta.path)
        return _html(
            request,
            "document.html",
            {
                "doc_id": doc_id,
                "missing": False,
                "unloaded": False,
                "q": q,
                "title": book or display_title(meta.title),
                "book": book,
                "body": text,
                "words": len(text.split()),
                "back": (
                    _search_href(q, k=10, scorer="bm25", mode="keyword", page=1)
                    if q
                    else "/"
                ),
            },
        )


def _render_search(
    request: Request,
    *,
    q: str,
    k: int,
    scorer: Literal["bm25", "tfidf"],
    mode: Literal["keyword", "semantic", "hybrid"],
    page: int,
    index: Index | None,
) -> HTMLResponse:
    error = ""
    total = 0
    took_ms = 0.0
    hits: list[dict[str, object]] = []
    status = 200
    query = q.strip()

    if query and index is None:
        status = 503
        error = "The index is not loaded yet."
    elif query and index is not None:
        try:
            result = search_api.run_search(
                index,
                query,
                k=k,
                scorer=scorer,
                page=page,
                mode=mode,
                embeddings=getattr(request.app.state, "embeddings", None),
                encode=_encoder(request) if mode != "keyword" else None,
            )
        except QuerySyntaxError as exc:
            status = 400
            error = str(exc)
        except EmbeddingsUnavailable as exc:
            status = 400
            error = str(exc)
        else:
            total = result.total
            took_ms = result.took_ms
            hits = [_hit_view(index, hit) for hit in result.results]

    pages = math.ceil(total / k) if total else 0

    def _href(target: int) -> str:
        return _search_href(query, k=k, scorer=scorer, mode=mode, page=target)

    context: dict[str, object] = {
        "q": query,
        "k": k,
        "scorer": scorer,
        "mode": mode,
        "page": page,
        "pages": pages,
        "window": _page_window(page, pages),
        "total": total,
        "took_ms": took_ms,
        "took_label": _format_ms(took_ms),
        "hits": hits,
        "error": error,
        "past_end": bool(total and page > pages),
        "ready": index is not None,
        "documents": index.num_docs if index is not None else 0,
        "vocabulary": len(index) if index is not None else 0,
        "suggestions": SUGGESTIONS,
        "href": _href,
    }
    name = "results.html" if request.headers.get("hx-request") else "search.html"
    return _html(request, name, context, status_code=status)


def _doc_meta(index: Index, doc_id: int) -> DocMeta | None:
    # ``Index`` is an untyped ``Mapping``, so the attribute is read as ``Any``.
    table = cast(dict[int, DocMeta], getattr(index, "doc_meta"))
    return table.get(doc_id)


def _hit_view(index: Index, hit: ApiSearchResult) -> dict[str, object]:
    meta = _doc_meta(index, hit.doc_id)
    path = meta.path if meta is not None else ""
    return {
        "doc_id": hit.doc_id,
        "title": display_title(hit.title),
        "score": f"{hit.score:.2f}",
        "snippet": hit.snippet,
        "book": book_name(path),
    }


def _lookup(index: Index, doc_id: int) -> tuple[DocMeta, str] | None:
    meta = _doc_meta(index, doc_id)
    if meta is None:
        return None
    return meta, index.document_text(doc_id)


def _html(
    request: Request,
    name: str,
    context: dict[str, object],
    *,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        name,
        context,
        status_code=status_code,
    )


def _index_bytes(settings: Settings) -> int | None:
    path = settings.index_path
    if path is None or not path.is_file():
        return None
    return path.stat().st_size


def _uptime(request: Request) -> float:
    started = getattr(request.app.state, "started_at", None)
    if not isinstance(started, float | int):
        return 0.0
    return round(time.monotonic() - float(started), 3)


def _format_ms(took_ms: float) -> str:
    if took_ms < 10:
        return f"{took_ms:.1f}"
    return f"{took_ms:.0f}"


def _search_href(q: str, *, k: int, scorer: str, mode: str, page: int) -> str:
    return "/?" + urlencode(
        {"q": q, "k": k, "scorer": scorer, "mode": mode, "page": page}
    )


def _page_window(page: int, pages: int) -> list[int | None]:
    """Page numbers with ``None`` standing in for an ellipsis."""
    if pages <= 1:
        return []
    if pages <= 7:
        return list(range(1, pages + 1))
    wanted = {1, pages, page - 1, page, page + 1}
    ordered = sorted(n for n in wanted if 1 <= n <= pages)
    window: list[int | None] = []
    previous = 0
    for number in ordered:
        if previous and number > previous + 1:
            window.append(None)
        window.append(number)
        previous = number
    return window


app = create_app()

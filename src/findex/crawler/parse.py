"""Pull a title, visible text, and absolute links out of one HTML page.

Parsing runs in a worker thread (``asyncio.to_thread``). It is CPU work, and a
blocking call on the event-loop thread would freeze every in-flight request.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from findex.crawler.urls import canonicalize

logger = logging.getLogger(__name__)


class _PageParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.base_url = page_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.hrefs: list[str] = []
        self._in_title = False
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in {"script", "style", "noscript"}:
            self._skip += 1
        if name == "title":
            self._in_title = True
        if name == "base":
            href = _attr(attrs, "href")
            if href:
                self.base_url = urljoin(self.page_url, href)
        if name in {"a", "area"}:
            href = _attr(attrs, "href")
            if href:
                self.hrefs.append(href)
        if name in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "section"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "title":
            self._in_title = False
        if name in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._skip or not data.strip():
            return
        self.text_parts.append(data)


def parse_html(page_url: str, html: str) -> tuple[str, str, tuple[str, ...]]:
    """Return ``(title, text, canonical_links)``.

    ``text`` starts with the title so ``findex index`` can use its first line.
    Links are absolute, canonical, and ``http(s)`` only.
    """
    parser = _PageParser(page_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Malformed markup still yields whatever was collected before the error.
        logger.info("stopped HTML parse early for %s", page_url)

    title = " ".join("".join(parser.title_parts).split())
    body = " ".join("".join(parser.text_parts).split())
    if title and body:
        text = f"{title}\n{body}"
    else:
        text = title or body

    links: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        absolute = urljoin(parser.base_url, href.strip())
        canon = canonicalize(absolute)
        if canon in seen:
            continue
        parts = urlsplit(canon)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            continue
        seen.add(canon)
        links.append(canon)
    if not title:
        title = page_url
    return title, text, tuple(links)


def is_html(content_type: str | None, body: bytes) -> bool:
    """True when the response should be parsed as a page.

    A missing ``Content-Type`` is treated as HTML. ``text/plain`` is parsed
    only when the body actually starts like a document.
    """
    if not content_type:
        return True
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime in {"text/html", "application/xhtml+xml"}:
        return True
    if mime == "text/plain":
        return _looks_like_html(body)
    return False


def _looks_like_html(body: bytes) -> bool:
    head = body.lstrip()[:200].lower()
    return head.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def _attr(attrs: list[tuple[str, str | None]], key: str) -> str | None:
    for name, value in attrs:
        if name.lower() == key and value:
            return value
    return None

"""Canonical URLs so the same page is not fetched twice."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlsplit, urlunsplit


def canonicalize(url: str) -> str:
    """Lowercase the scheme and host, drop the fragment, normalize the path and query.

    Trailing slashes collapse (``/docs/`` and ``/docs`` are one page; the root stays
    ``/``). Query pairs are sorted so ``?b=1&a=2`` matches ``?a=2&b=1``. Default
    ports are omitted. A URL we cannot split is returned stripped, unchanged.
    """
    raw = url.strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in {"http", "https"} or not host:
        return raw

    port = parts.port
    default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    if port is None or default:
        netloc = host
    else:
        netloc = f"{host}:{port}"

    path = _normalize_path(parts.path)
    query = _normalize_query(parts.query)
    return urlunsplit((scheme, netloc, path, query, ""))


def origin_of(url: str) -> str:
    """``scheme://host[:port]`` for the robots.txt cache key."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def hostname_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _normalize_path(path: str) -> str:
    if path in {"", "/"}:
        return "/"
    segments: list[str] = []
    for segment in path.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    normalized = "/" + "/".join(segments)
    if len(normalized) > 1 and normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def _normalize_query(query: str) -> str:
    if not query:
        return ""
    pairs = parse_qsl(query, keep_blank_values=True)
    pairs.sort()
    # Keep a stable encoding without importing urlencode's doseq surprises:
    # each pair is already decoded by parse_qsl, so rebuild with quote_plus.
    from urllib.parse import urlencode

    return urlencode(pairs)

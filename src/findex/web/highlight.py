"""Turn snippet markers and query terms into escaped HTML."""

from __future__ import annotations

import re
from pathlib import Path

from markupsafe import Markup, escape

from findex.tokenize import tokenize

_SNIPPET_MARK = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

# Folder slugs from scripts/build_corpus.py. Anything else (the Docker
# sample JSONL lives in ``corpus/``) falls back to the stored title.
BOOKS: dict[str, str] = {
    "pride-and-prejudice": "Pride and Prejudice",
    "alice-in-wonderland": "Alice's Adventures in Wonderland",
    "frankenstein": "Frankenstein",
    "sherlock-holmes": "The Adventures of Sherlock Holmes",
    "dorian-gray": "The Picture of Dorian Gray",
    "tale-of-two-cities": "A Tale of Two Cities",
    "huckleberry-finn": "Adventures of Huckleberry Finn",
    "dracula": "Dracula",
}

_SKIP_TERMS = frozenset({"and", "or", "not"})


def book_name(path: str) -> str | None:
    slug = Path(path.split("#", 1)[0]).parent.name
    return BOOKS.get(slug)


def highlight_snippet(snippet: str) -> Markup:
    """``make_snippet`` wraps hits in ``**term**``. Escape first, then mark."""
    escaped = str(escape(snippet))
    html = _SNIPPET_MARK.sub(r"<mark>\1</mark>", escaped)
    return Markup(html)


def highlight_text(text: str, query: str) -> Markup:
    """Escape a full passage, then highlight query terms."""
    escaped = str(escape(text))
    terms = sorted(
        {
            token
            for token in tokenize(query)
            if token not in _SKIP_TERMS and len(token) > 1
        },
        key=len,
        reverse=True,
    )
    if not terms:
        return Markup(escaped)
    pattern = "|".join(re.escape(term) for term in terms)
    html = re.sub(
        rf"\b(?:{pattern})\b",
        lambda match: f"<mark>{match.group(0)}</mark>",
        escaped,
        flags=re.IGNORECASE,
    )
    return Markup(html)


def display_title(title: str) -> str:
    """Drop a mid-word cut. Stored titles are clipped at 80 characters."""
    if len(title) >= 80 and " " in title:
        return title.rsplit(" ", 1)[0].rstrip(".,;:") + "…"
    return title


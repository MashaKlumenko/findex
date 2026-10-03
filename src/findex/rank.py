"""TF-IDF / BM25 ranking, snippets, and a swappable Scorer protocol."""

from __future__ import annotations

import heapq
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Protocol, runtime_checkable, Literal

from findex.index import Index, Posting
from findex.query import match_doc_ids, parse
from findex.timing import timed
from findex.tokenize import TOKEN_RE

# Новий синтаксис PEP 695 для аліасів типів
DocId = int


@runtime_checkable
class Scorer(Protocol):
    """Duck-typed scorer: TF-IDF and BM25 are interchangeable."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    """``tf * log(N / df)`` — the textbook baseline."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        tf = posting.tf
        df = index.df(term)
        n = index.num_docs
        if tf <= 0 or df <= 0 or n <= 0:
            return 0.0
        return tf * math.log(n / df)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)

    def __repr__(self) -> str:
        return "TfIdf()"


class BM25:
    """Okapi BM25 with Lucene-style IDF, default ``k1=1.5``, ``b=0.75``."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, term: str, posting: Posting, index: Index) -> float:
        tf = posting.tf
        df = index.df(term)
        n = index.num_docs
        if tf <= 0 or df <= 0 or n <= 0:
            return 0.0
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = index.avg_doc_length
        if avgdl <= 0:
            return 0.0
        dl = index.doc_length(posting.doc_id)
        k1, b = self.k1, self.b
        denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
        return idf * (tf * (k1 + 1.0)) / denom

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def get_scorer(name: Literal["tfidf", "bm25"] | str) -> Scorer:
    """Get scorer instance by name. Strict Literal limits are added for CLI integration."""
    key = name.strip().lower()
    if key in {"tfidf", "tf-idf", "tf_idf"}:
        return TfIdf()
    if key == "bm25":
        return BM25()
    raise ValueError(f"unknown scorer {name!r}")


@dataclass(order=True)
class SearchResult:
    """Ranked hit. ``order=True`` so ``sorted(results)`` is by score."""

    doc_id: DocId = field(compare=False)
    score: float
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)

    def __str__(self) -> str:
        line = f"{self.score:8.4f}  {self.doc_id}  {self.title}"
        if self.snippet:
            return f"{line}\n          {self.snippet}"
        return line


def make_snippet(text: str, terms: Sequence[str], *, radius: int = 80) -> str:
    """±80 characters around the densest query-term window; ``**term**`` highlights."""
    folded = unicodedata.normalize("NFC", text).casefold()
    termset = {t.casefold() for t in terms if t}
    matches = [
        (m.start(), m.end())
        for m in TOKEN_RE.finditer(folded)
        if m.group(0) in termset
    ]
    if not matches:
        clip = folded[: radius * 2].strip()
        return clip + ("…" if len(folded) > radius * 2 else "")

    best_start, best_count = matches[0][0], -1
    for start, _end in matches:
        window_end = start + 2 * radius
        count = sum(1 for a, b in matches if a >= start and b <= window_end)
        if count > best_count:
            best_count = count
            best_start = start

    lo = max(0, best_start - radius)
    hi = min(len(folded), best_start + radius)
    window = folded[lo:hi]
    # Highlight longest terms first so "event" does not eat "event loop" pieces.
    highlighted = window
    for term in sorted(termset, key=len, reverse=True):
        highlighted = re.sub(
            rf"\b({re.escape(term)})\b",
            r"**\1**",
            highlighted,
            flags=re.IGNORECASE,
        )
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(folded) else ""
    return f"{prefix}{highlighted}{suffix}"


def _positive_terms(query: str) -> list[str]:
    try:
        return list(parse(query).positive_terms())
    except Exception:
        return []


@timed
def ranked_search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    snippets: bool = True,
) -> list[SearchResult]:
    """Accumulate per-document scores, then ``heapq.nlargest`` for top-k."""
    if scorer is None:
        scorer = BM25()
    if not query.strip():
        return []
    doc_ids = match_doc_ids(index, query)
    if not doc_ids:
        return []
    allowed = set(doc_ids)
    terms = _positive_terms(query)
    scores: dict[DocId, float] = defaultdict(float)
    for term in terms:
        for posting in index.iter_postings(term):
            if posting.doc_id in allowed:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    if not scores:
        # Phrase/boolean match but no positive terms (pure NOT) — keep ids, zero score.
        scored_items = [(doc_id, 0.0) for doc_id in doc_ids]
    else:
        scored_items = list(scores.items())
    top = heapq.nlargest(k, scored_items, key=itemgetter(1))
    results: list[SearchResult] = []
    for doc_id, score in top:
        meta = index.doc_meta.get(doc_id)
        title = meta.title if meta is not None else "?"
        snippet = ""
        if snippets:
            snippet = make_snippet(index.document_text(doc_id), terms)
        results.append(SearchResult(doc_id=doc_id, score=score, title=title, snippet=snippet))
    return results

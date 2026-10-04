"""findex — a search engine built one Python idiom at a time."""

from findex.corpus import Document, iter_documents
from findex.index import DocMeta, Index, Posting, build_index, build_partial, merge
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.rank import BM25, SearchResult, TfIdf, ranked_search
from findex.search import boolean_search, search  # type: ignore
from findex.store import load, open_index, save
from findex.tokenize import tokenize

__all__ = [
    "And",
    "BM25",
    "DocMeta",
    "Document",
    "Index",
    "Not",
    "Or",
    "Phrase",
    "Posting",
    "SearchResult",
    "Term",
    "TfIdf",
    "boolean_search",
    "build_index",
    "build_partial",
    "iter_documents",
    "merge",
    "load",
    "open_index",
    "parse",
    "ranked_search",
    "save",
    "search",
    "tokenize",
]

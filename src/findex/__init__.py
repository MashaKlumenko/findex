"""findex — a search engine built one Python idiom at a time."""

from findex.corpus import Document, iter_documents
from findex.index import DocMeta, Index, Posting, build_index
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
    "iter_documents",
    "load",
    "open_index",
    "parse",
    "ranked_search",
    "save",
    "search",
    "tokenize",
]

"""findex — a search engine built one Python idiom at a time."""

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

__all__ = ["Document", "iter_documents", "tokenize"]

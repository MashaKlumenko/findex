"""Recursive-descent Boolean query language.

Grammar (OR binds looser than AND; AND is implicit between adjacent clauses)::

    or_expr  → and_expr ('OR' and_expr)*
    and_expr → not_expr ('AND'? not_expr)*
    not_expr → 'NOT' atom | atom
    atom     → TERM | PHRASE | '(' or_expr ')'

So ``a OR b c`` is ``Or(Term('a'), And(Term('b'), Term('c')))``.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

from findex.timing import timed
from findex.tokenize import TOKEN_RE, tokenize

if TYPE_CHECKING:
    from findex.index import Index

logger = logging.getLogger(__name__)

# Live indexes by id, so lru_cache keys stay hashable (ints + strings).
_LIVE_INDEXES: WeakValueDictionary[int, Index] = WeakValueDictionary()


class QuerySyntaxError(ValueError):
    """The query string is not a valid Boolean expression."""


class QueryNode:
    """Mixin: queries compose with ``&``, ``|``, ``~`` like sets."""

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index, *, engine: str = "merge") -> list[int]:
        raise NotImplementedError

    def positive_terms(self) -> list[str]:
        """Terms that should contribute to ranking (everything not under NOT)."""
        raise NotImplementedError


def _ops(engine: str):
    from findex.search import merge_and, merge_not, merge_or, set_and, set_not, set_or

    if engine == "set":
        return set_and, set_or, set_not
    if engine == "merge":
        return merge_and, merge_or, merge_not
    raise ValueError(f"unknown engine {engine!r}")


def _universe(index: Index) -> list[int]:
    return list(range(index.num_docs))


@dataclass(frozen=True)
class Term(QueryNode):
    term: str

    def evaluate(self, index: Index, *, engine: str = "merge") -> list[int]:
        return index.doc_ids_for(self.term)

    def positive_terms(self) -> list[str]:
        return [self.term]


@dataclass(frozen=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index, *, engine: str = "merge") -> list[int]:
        if not self.terms:
            return []
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index, engine=engine)
        if not index.store_positions:
            raise ValueError("phrase queries require an index built with --positions")
        and_op, _, _ = _ops(engine)
        docs: list[int] | None = None
        for term in self.terms:
            ids = index.doc_ids_for(term)
            docs = ids if docs is None else and_op(docs, ids)
        if not docs:
            return []
        return [doc_id for doc_id in docs if _phrase_in_doc(index, self.terms, doc_id)]

    def positive_terms(self) -> list[str]:
        return list(self.terms)


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index, *, engine: str = "merge") -> list[int]:
        and_op, _, _ = _ops(engine)
        return and_op(
            self.left.evaluate(index, engine=engine),
            self.right.evaluate(index, engine=engine),
        )

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index, *, engine: str = "merge") -> list[int]:
        _, or_op, _ = _ops(engine)
        return or_op(
            self.left.evaluate(index, engine=engine),
            self.right.evaluate(index, engine=engine),
        )

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Not(QueryNode):
    child: QueryNode

    def evaluate(self, index: Index, *, engine: str = "merge") -> list[int]:
        _, _, not_op = _ops(engine)
        return not_op(_universe(index), self.child.evaluate(index, engine=engine))

    def positive_terms(self) -> list[str]:
        return []


@dataclass(frozen=True)
class _Tok:
    kind: str
    value: object = None


def _lex(query: str) -> list[_Tok]:
    text = unicodedata.normalize("NFC", query).casefold()
    tokens: list[_Tok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(_Tok("LPAREN"))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Tok("RPAREN"))
            i += 1
            continue
        if ch == '"':
            end = text.find('"', i + 1)
            if end < 0:
                raise QuerySyntaxError("unterminated phrase")
            inner = tuple(tokenize(text[i + 1 : end]))
            tokens.append(_Tok("PHRASE", inner))
            i = end + 1
            continue
        match = TOKEN_RE.match(text, i)
        if not match:
            raise QuerySyntaxError(f"unexpected character {ch!r} at {i}")
        word = match.group(0)
        i = match.end()
        if word == "and":
            tokens.append(_Tok("AND"))
        elif word == "or":
            tokens.append(_Tok("OR"))
        elif word == "not":
            tokens.append(_Tok("NOT"))
        else:
            tokens.append(_Tok("TERM", word))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Tok]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> _Tok | None:
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return None

    def take(self, kind: str) -> _Tok:
        tok = self.peek()
        if tok is None or tok.kind != kind:
            raise QuerySyntaxError(f"expected {kind}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> QueryNode:
        node = self.or_expr()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected token {self.peek()}")
        return node

    def or_expr(self) -> QueryNode:
        node = self.and_expr()
        while self.peek() is not None and self.peek().kind == "OR":
            self.i += 1
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> QueryNode:
        node = self.not_expr()
        while self.peek() is not None and self.peek().kind not in {"OR", "RPAREN"}:
            if self.peek().kind == "AND":
                self.i += 1
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> QueryNode:
        if self.peek() is not None and self.peek().kind == "NOT":
            self.i += 1
            return Not(self.atom())
        return self.atom()

    def atom(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("unexpected end of query")
        if tok.kind == "LPAREN":
            self.i += 1
            node = self.or_expr()
            self.take("RPAREN")
            return node
        if tok.kind == "PHRASE":
            self.i += 1
            return Phrase(tok.value)  # type: ignore[arg-type]
        if tok.kind == "TERM":
            self.i += 1
            return Term(tok.value)  # type: ignore[arg-type]
        raise QuerySyntaxError(f"unexpected token {tok}")


def parse(query: str) -> QueryNode:
    tokens = _lex(query)
    if not tokens:
        raise QuerySyntaxError("empty query")
    return _Parser(tokens).parse()


def register_index(index: Index) -> None:
    _LIVE_INDEXES[id(index)] = index


def unregister_index(index: Index) -> None:
    _LIVE_INDEXES.pop(id(index), None)
    _match_doc_ids_cached.cache_clear()


def _phrase_in_doc(index: Index, terms: tuple[str, ...], doc_id: int) -> bool:
    pos_lists = [index.positions(term, doc_id) for term in terms]
    if any(not plist for plist in pos_lists):
        return False
    first = pos_lists[0]
    rest = pos_lists[1:]
    for start in first:
        if all((start + offset) in plist for offset, plist in enumerate(rest, start=1)):
            return True
    return False


def _match_doc_ids_uncached(index_id: int, query: str) -> tuple[int, ...]:
    index = _LIVE_INDEXES[index_id]
    return tuple(parse(query).evaluate(index))


_match_doc_ids_cached = lru_cache(maxsize=256)(_match_doc_ids_uncached)
_match_doc_ids_timed = timed(_match_doc_ids_cached)


def match_doc_ids(index: Index, query: str) -> list[int]:
    """Parse + evaluate, memoized by ``(id(index), query)``.

    A repeated query is a cache hit: ``@timed`` logs ~0 ms and this function
    logs ``lru_cache hit``.
    """
    register_index(index)
    before = _match_doc_ids_cached.cache_info()
    ids = _match_doc_ids_timed(id(index), query)
    after = _match_doc_ids_cached.cache_info()
    if after.hits > before.hits:
        logger.info(
            "lru_cache hit for query %r (hits=%d misses=%d)",
            query,
            after.hits,
            after.misses,
        )
    else:
        logger.info(
            "lru_cache miss for query %r (hits=%d misses=%d)",
            query,
            after.hits,
            after.misses,
        )
    return list(ids)


def cache_info():
    return _match_doc_ids_cached.cache_info()

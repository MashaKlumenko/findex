"""Parser tests compare trees, not strings."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse


def test_or_binds_looser_than_implicit_and() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_explicit_and_parentheses_not_and_phrase() -> None:
    tree = parse('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert tree == expected


def test_operator_overloading() -> None:
    assert Term("python") & (Term("async") | Term("await")) == And(
        Term("python"),
        Or(Term("async"), Term("await")),
    )
    assert ~Term("java") == Not(Term("java"))


def test_not_and_empty() -> None:
    assert parse("NOT cat") == Not(Term("cat"))
    with pytest.raises(QuerySyntaxError):
        parse("")
    with pytest.raises(QuerySyntaxError):
        parse("(a OR b")


def test_phrase_needs_positions(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("event loop async", encoding="utf-8")
    (tmp_path / "b.txt").write_text("loop event", encoding="utf-8")
    bare = build_index(iter_documents(tmp_path))
    with pytest.raises(ValueError, match="positions"):
        parse('"event loop"').evaluate(bare)
    with_pos = build_index(iter_documents(tmp_path), positions=True)
    assert parse('"event loop"').evaluate(with_pos) == [0]
    assert parse('"loop event"').evaluate(with_pos) == [1]

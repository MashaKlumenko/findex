"""Assert-style checks for the streaming tokenizer (pytest will pick these up later)."""

from __future__ import annotations

import inspect
import unicodedata

from findex.tokenize import tokenize


def tokens(text: str) -> list[str]:
    return list(tokenize(text))


def test_tokenize_is_a_generator() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    gen = tokenize("hello")
    assert inspect.isgenerator(gen)
    assert next(gen) == "hello"


def test_mixed_case() -> None:
    assert tokens("The QUICK Brown Fox") == ["the", "quick", "brown", "fox"]


def test_cyrillic() -> None:
    assert tokens("Привіт, Київ!") == ["привіт", "київ"]
    assert tokens("п'ять яблук") == ["п'ять", "яблук"]


def test_combining_mark_accent() -> None:
    composed = "café"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    assert unicodedata.normalize("NFC", decomposed) == composed
    assert tokens(composed) == ["café"]
    assert tokens(decomposed) == ["café"]


def test_punctuation_is_stripped() -> None:
    assert tokens("Hello, world!") == ["hello", "world"]
    assert tokens("(well-known) — really?") == ["well-known", "really"]


def test_empty_string() -> None:
    assert tokens("") == []
    assert tokens("   \n\t") == []


def test_apostrophes_and_hyphens() -> None:
    assert tokens("don't stop") == ["don't", "stop"]
    assert tokens("state-of-the-art") == ["state-of-the-art"]


def test_digits_are_kept() -> None:
    assert tokens("Python 3.11 and utf-8") == ["python", "3", "11", "and", "utf-8"]


def test_casefold_eszett() -> None:
    # lower() would keep ß; casefold maps it to ss
    assert tokens("Straße") == ["strasse"]

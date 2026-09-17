"""Streaming tokenizer: NFC + casefold, then a lazy regex over words.

Policy (also in the README)
---------------------------
Apostrophes
    Internal apostrophes stay inside a token so ``don't`` and Ukrainian
    ``п'ять`` remain one word. Leading/trailing apostrophes are not part of
    ``\\w``, so they fall off with other punctuation.

Hyphens
    A hyphen between word characters is kept (``well-known``, ``state-of-the-art``
    as a sequence of hyphen-joined pieces: ``well-known`` is one token because
    the pattern allows one or more ``\\w+`` groups joined by ``-`` or ``'``).

Digits
    Digit runs are tokens. ``utf-8`` becomes ``utf-8``; a bare ``42`` is kept.
    Search engines usually want numbers (years, versions, issue ids).

``re.finditer`` is used instead of ``findall`` so a huge document does not
become a giant list of strings before the caller asks for them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# Word characters (Unicode-aware \\w) joined by internal apostrophes or hyphens.
# Examples: don't, п'ять, well-known, utf-8, Київ
TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Yield word tokens from ``text``, one at a time.

    1. NFC-normalize so ``café`` and ``cafe\\u0301`` become the same string.
    2. ``casefold`` for caseless matching (stronger than ``lower``).
    3. Stream matches with ``re.finditer``.
    """
    if not text:
        return
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0)

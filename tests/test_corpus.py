"""Sanity checks that iter_documents is a generator and streams files."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from findex.corpus import iter_documents


def test_iter_documents_is_a_generator() -> None:
    assert inspect.isgeneratorfunction(iter_documents)


def test_txt_tree(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.md").write_text("світе", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    docs = list(iter_documents(tmp_path))
    ids = {d.doc_id for d in docs}
    texts = {d.text for d in docs}
    assert ids == {"a.txt", "sub/b.md"}
    assert texts == {"hello", "світе"}


def test_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    rows = [
        {"id": "g1", "text": "first"},
        {"doc_id": "g2", "body": "second"},
        "not-json",
        {"text": "third"},
    ]
    path.write_text(
        "\n".join(json.dumps(r) if not isinstance(r, str) else r for r in rows),
        encoding="utf-8",
    )
    docs = list(iter_documents(path))
    assert [d.text for d in docs] == ["first", "second", "third"]
    assert [d.doc_id for d in docs[:2]] == ["g1", "g2"]
    assert docs[2].doc_id.endswith(":4")

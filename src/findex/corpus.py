"""Lazy document stream over a directory of text files or a JSON-lines file.

``iter_documents`` is a generator: it opens one file at a time, yields a
``Document``, and moves on. The corpus never sits in memory as a list.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}
JSONL_SUFFIXES = {".jsonl", ".jsonl.gz"}  # .gz handled in a later stretch


@dataclass(frozen=True, slots=True)
class Document:
    """One unit of the corpus.

    ``doc_id`` is a stable, filesystem-relative identifier (posix path),
    not a list index — so it stays meaningful after we persist an index.
    """

    doc_id: str
    path: Path
    text: str
    # 1-based line in a JSONL file. Snippets re-read that line, not the whole file.
    source_line: int | None = None


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents from ``root`` one at a time.

    * If ``root`` is a ``.jsonl`` file, each line is one document.
    * If ``root`` is a directory, walk it with ``rglob`` for ``.txt`` / ``.md``
      files, and also any ``.jsonl`` files found in the tree.
    * Undecodable bytes are replaced and logged; a missing/unreadable file
      is skipped rather than aborting the whole stream.
    """
    root = Path(root)
    if root.is_file():
        yield from _iter_file(root, id_base=root.parent)
        return
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root does not exist: {root}")

    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES.union({".jsonl"})
    )
    for path in paths:
        yield from _iter_file(path, id_base=root)


def _iter_file(path: Path, id_base: Path) -> Iterator[Document]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path, id_base)
        return
    text = _read_text(path)
    if text is None:
        return
    yield Document(doc_id=_doc_id(path, id_base), path=path, text=text)


def _iter_jsonl(path: Path, id_base: Path) -> Iterator[Document]:
    try:
        fh = path.open(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("skipping unreadable jsonl %s: %s", path, exc)
        return

    with fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping bad json in %s:%s: %s", path, line_no, exc)
                continue
            text = _text_from_json(obj)
            if not text:
                logger.warning("skipping empty json document in %s:%s", path, line_no)
                continue
            doc_id = (
                obj.get("id")
                or obj.get("doc_id")
                or f"{_doc_id(path, id_base)}:{line_no}"
            )
            yield Document(
                doc_id=str(doc_id), path=path, text=text, source_line=line_no
            )


def _text_from_json(obj: object) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "body", "content"):
            value = obj.get(key)
            if isinstance(value, str):
                return value
    return ""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("skipping unreadable file %s: %s", path, exc)
        return None


def _doc_id(path: Path, id_base: Path) -> str:
    try:
        return path.relative_to(id_base).as_posix()
    except ValueError:
        return path.as_posix()


def list_corpus_paths(root: Path) -> list[Path]:
    """Files ``iter_documents`` would open, in the same order.

    Workers receive these paths and read the bytes themselves. A ``.txt`` /
    ``.md`` file is one document; a ``.jsonl`` file may be many.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root does not exist: {root}")
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES.union({".jsonl"})
    )


def iter_paths(paths: Iterable[Path]) -> Iterator[Document]:
    """Yield documents from already-chosen files, in list order."""
    for path in paths:
        path = Path(path)
        yield from _iter_file(path, id_base=path.parent)


def count_documents(path: Path) -> int:
    """How many documents ``path`` contributes, without keeping the text.

    ``.txt`` / ``.md`` are one document. ``.jsonl`` is counted by walking
    lines the same way ``iter_documents`` does, so bad lines are skipped.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return 1
    if suffix == ".jsonl":
        return sum(1 for _ in _iter_file(path, id_base=path.parent))
    return 0

"""Build an inverted index in one pass over a lazily streamed corpus.

    python -m findex.index data/ --out data/index.pkl
    python -m findex.index data/ --out data/index.json --positions
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.stats import format_bytes
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

# slots | plain | array — the three rows of the Lab 2 memory table.
Representation = str


@dataclass(frozen=True, slots=True)
class Posting:
    """One (document, term-frequency) pair, optionally with token offsets."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlainPosting:
    """Same fields as ``Posting``, but with a per-instance ``__dict__``.

    Used only for the memory-study row that shows what ``slots=True`` buys.
    """

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass(slots=True)
class Index:
    """In-memory inverted index. Postings lists are sorted by ``doc_id``."""

    postings: dict
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    store_positions: bool = False
    representation: str = "slots"

    def n_docs(self) -> int:
        return len(self.doc_meta)

    def df(self) -> int:
        # Цей метод залишається без змін
        return len(self.doc_meta)

    def df_term(self, term: str) -> int:  # Перейменуємо або залишимо df
        plist = self.postings.get(term)
        if not plist:
            return 0
        if self.representation == "array":
            # Рахуємо кількість унікальних doc_id у пласкому масиві
            count = 0
            i = 0
            while i < len(plist):
                count += 1
                tf = plist[i + 1]
                # Пропускаємо doc_id, tf та всі позиції цього документа
                i += 2 + (tf if self.store_positions else 0)
            return count
        return len(plist)

    def doc_ids_for(self, term: str) -> list[int]:
        """Sorted document ids that contain ``term`` (empty if unknown)."""
        plist = self.postings.get(term)
        if not plist:
            return []
        if self.representation == "array":
            doc_ids = []
            i = 0
            while i < len(plist):
                doc_ids.append(plist[i])
                tf = plist[i + 1]
                i += 2 + (tf if self.store_positions else 0)
            return doc_ids
        return [p.doc_id for p in plist]


def _title_from_document(doc: Document) -> str:
    for line in doc.text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return Path(doc.doc_id).stem or str(doc.path)


def _term_stats(
    text: str, *, positions: bool
) -> tuple[int, dict[str, tuple[int, tuple[int, ...]]]]:
    """Return ``(n_tokens, term -> (tf, positions))`` for one document."""
    if positions:
        posmap: dict[str, list[int]] = defaultdict(list)
        n = 0
        for pos, tok in enumerate(tokenize(text)):
            posmap[tok].append(pos)
            n += 1
        packed = {
            term: (len(poslist), tuple(poslist)) for term, poslist in posmap.items()
        }
        return n, packed
    counts = Counter(tokenize(text))
    n = sum(counts.values())
    return n, {term: (tf, ()) for term, tf in counts.items()}


def build_index(
    docs: Iterable[Document],
    *,
    positions: bool = False,
    representation: Representation = "slots",
) -> Index:
    """Consume the Lab 1 pipeline and produce sorted postings.

    Integer ``doc_id``s are assigned in stream order so a later merge can
    walk lists with two pointers. Path/title stay in ``doc_meta``.
    """
    if representation not in {"slots", "plain", "array"}:
        raise ValueError(f"unknown representation: {representation!r}")

    # defaultdict(list) is the "no if term not in index" build.
    # The array path appends raw uint32s so we never allocate a Posting.
    object_bucket: dict[str, list] = defaultdict(list)
    array_ids: dict[str, array] = defaultdict(lambda: array("I"))
    array_tfs: dict[str, array] = defaultdict(lambda: array("I"))
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):
        n_tokens, stats = _term_stats(doc.text, positions=positions)
        doc_lengths[doc_id] = n_tokens
        doc_meta[doc_id] = DocMeta(path=str(doc.path), title=_title_from_document(doc))
        for term, (tf, pos) in stats.items():
            if representation == "array":
                array_ids[term].append(doc_id)
                array_tfs[term].append(tf)
            elif representation == "plain":
                object_bucket[term].append(PlainPosting(doc_id, tf, pos))
            else:
                object_bucket[term].append(Posting(doc_id, tf, pos))

    if representation == "array":
        postings: dict = {
            term: (array_ids[term], array_tfs[term]) for term in array_ids
        }
    else:
        postings = {}
        for term, rows in object_bucket.items():
            rows.sort(key=lambda item: item.doc_id)
            postings[term] = rows

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        store_positions=positions,
        representation=representation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an inverted index from a corpus directory.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="directory of .txt/.md/.jsonl files, or a single .jsonl file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output path; suffix chooses format (.pkl/.bin pickle, .json JSON)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only the first N documents (islice on the lazy stream)",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="record token offsets per posting (larger index, phrase-ready)",
    )
    parser.add_argument(
        "--representation",
        choices=("slots", "plain", "array"),
        default="slots",
        help="postings storage used for the Lab 2 memory table",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    docs: Iterable[Document] = iter_documents(args.root)
    if args.limit is not None:
        docs = islice(docs, args.limit)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(
        docs,
        positions=args.positions,
        representation=args.representation,
    )
    build_elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    from findex.store import save

    t1 = time.perf_counter()
    save(index, args.out)
    save_elapsed = time.perf_counter() - t1
    size = args.out.stat().st_size

    print(f"documents:      {index.n_docs()}")
    print(f"vocabulary:     {len(index.postings)}")
    print(f"tokens:         {sum(index.doc_lengths.values())}")
    print(f"representation: {index.representation}")
    print(f"positions:      {index.store_positions}")
    print(f"build elapsed:  {build_elapsed:.3f} s")
    print(f"peak memory:    {format_bytes(peak)} ({peak} bytes)")
    print(f"save elapsed:   {save_elapsed:.3f} s")
    print(f"file size:      {format_bytes(size)} ({size} bytes)")
    print(f"wrote:          {args.out}")
    return 0


if __name__ == "__main__":
    # Re-import so pickle records Posting/Index as findex.index.*, not __main__.
    from findex.index import main as _main

    sys.exit(_main())

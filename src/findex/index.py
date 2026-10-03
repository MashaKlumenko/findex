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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.stats import format_bytes
from findex.timing import timed
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


class Index(Mapping):
    """Inverted index that behaves like a ``Mapping[str, postings]``.

    ``len(index)`` is vocabulary size; ``"python" in index`` tests the term;
    ``index["python"]`` is the postings list (``KeyError`` if absent).
    """

    def __init__(
        self,
        postings: dict,
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        store_positions: bool = False,
        representation: str = "slots",
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.store_positions = store_positions
        self.representation = representation
        self._closed = False
        from findex.query import register_index

        register_index(self)

    def __getstate__(self) -> dict:
        return {
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "doc_meta": self.doc_meta,
            "store_positions": self.store_positions,
            "representation": self.representation,
        }

    def __setstate__(self, state: dict) -> None:
        self.postings = state["postings"]
        self.doc_lengths = state["doc_lengths"]
        self.doc_meta = state["doc_meta"]
        self.store_positions = state["store_positions"]
        self.representation = state["representation"]
        self._closed = False
        from findex.query import register_index

        register_index(self)

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str):
        return self.postings[term]

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    def n_docs(self) -> int:
        """Lab 2 alias for ``num_docs``."""
        return self.num_docs

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Document frequency of ``term`` (0 if unknown)."""
        plist = self.postings.get(term)
        if not plist:
            return 0
        if self.representation == "array":
            return len(plist[0])
        return len(plist)

    def df_term(self, term: str) -> int:
        return self.df(term)

    def doc_ids_for(self, term: str) -> list[int]:
        """Sorted document ids that contain ``term`` (empty if unknown)."""
        return [p.doc_id for p in self.iter_postings(term)]

    def iter_postings(self, term: str) -> Iterator[Posting]:
        plist = self.postings.get(term)
        if not plist:
            return
        if self.representation == "array":
            ids, tfs = plist
            for doc_id, tf in zip(ids, tfs, strict=False):
                yield Posting(int(doc_id), int(tf), ())
            return
        yield from plist

    def positions(self, term: str, doc_id: int) -> tuple[int, ...]:
        for posting in self.iter_postings(term):
            if posting.doc_id == doc_id:
                return posting.positions
        return ()

    def document_text(self, doc_id: int) -> str:
        meta = self.doc_meta.get(doc_id)
        if meta is None:
            return ""
        path = Path(meta.path)
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return meta.title

    def close(self) -> None:
        """Drop postings so a ``with open_index`` block always frees RAM."""
        if self._closed:
            return
        from findex.query import unregister_index

        unregister_index(self)
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_meta.clear()
        self.__dict__.pop("avg_doc_length", None)
        self._closed = True

    def __enter__(self) -> Index:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


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


@timed
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

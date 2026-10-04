"""Build an inverted index in one pass over a lazily streamed corpus.

    python -m findex.index data/ --out data/index.pkl
    python -m findex.index data/ --out data/index.json --positions
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from heapq import merge as heap_merge
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import numpy as np

from findex.corpus import (
    TEXT_SUFFIXES,
    Document,
    count_documents,
    iter_paths,
    list_corpus_paths,
)
from findex.resources import ResourceMonitor, gil_enabled
from findex.stats import format_bytes
from findex.timing import timed
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

# slots | plain | array — the three rows of the Lab 2 memory table.
Representation = str
ExecutorName = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[str, ...] = ("serial", "threads", "processes")
# Measured on this corpus: a serial build is about 1.3 s, and the pickle /
# process-startup tax is larger than the parallel gain. Serial stays the
# default. See Lab 5.
DEFAULT_EXECUTOR: ExecutorName = "serial"


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
        # Intern so a string that crossed a process boundary is the same
        # object as the ``"array"`` literal. Otherwise pickle memos diverge
        # and two equal indexes are not byte-identical.
        self.representation = sys.intern(representation)
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
        self.representation = sys.intern(state["representation"])
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

    @cached_property
    def doc_len_array(self) -> np.ndarray:
        """One ``int32`` length per document id. Index ``i`` is document ``i``."""
        if not self.doc_meta:
            return np.zeros(0, dtype=np.int32)
        size = max(self.doc_meta) + 1
        out = np.zeros(size, dtype=np.int32)
        for doc_id, length in self.doc_lengths.items():
            out[doc_id] = length
        return out

    def df(self, term: str) -> int:
        """Document frequency of ``term`` (0 if unknown)."""
        plist = self.postings.get(term)
        if not plist:
            return 0
        if _is_buffer(self.representation):
            return len(plist[0])
        return len(plist)

    def df_term(self, term: str) -> int:
        return self.df(term)

    def doc_ids_for(self, term: str) -> list[int]:
        """Sorted document ids that contain ``term`` (empty if unknown)."""
        pair = self.posting_arrays(term)
        if pair is None:
            return []
        return [int(doc_id) for doc_id in pair[0]]

    def posting_arrays(self, term: str) -> tuple[np.ndarray, np.ndarray] | None:
        """``int32`` doc ids and ``int32`` term frequencies for ``term``.

        ``array('I')`` postings are a ``np.frombuffer`` view (no copy) when
        every value fits in ``int32``. Object postings are copied once and
        cached. ``representation="numpy"`` already stores the arrays.
        """
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = self.__dict__.setdefault(
            "_posting_arrays", {}
        )
        cached = cache.get(term)
        if cached is not None:
            return cached
        built = _posting_arrays(self, term)
        if built is not None:
            cache[term] = built
        return built

    def iter_postings(self, term: str) -> Iterator[Posting]:
        plist = self.postings.get(term)
        if not plist:
            return
        if _is_buffer(self.representation):
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
        file_path, line_no = _jsonl_line(meta.path)
        if line_no is not None:
            text = _read_jsonl_line(file_path, line_no)
            return text or meta.title
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
        self.__dict__.pop("_posting_arrays", None)
        self.__dict__.pop("doc_len_array", None)
        self.__dict__.pop("avg_doc_length", None)
        self._closed = True

    def __enter__(self) -> Index:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _jsonl_line(path: str) -> tuple[str, int | None]:
    """``file.jsonl#12`` points at one document. Anything else is a whole file."""
    file, sep, tail = path.rpartition("#")
    if not sep or not file.lower().endswith(".jsonl"):
        return path, None
    try:
        line_no = int(tail)
    except ValueError:
        return path, None
    if line_no < 1:
        return path, None
    return file, line_no


def _read_jsonl_line(path: str, line_no: int) -> str:
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if number != line_no:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    for key in ("text", "body", "content"):
                        value = obj.get(key)
                        if isinstance(value, str):
                            return value
                if isinstance(obj, str):
                    return obj
                return line.strip()
    except (OSError, json.JSONDecodeError):
        return ""
    return ""


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


@dataclass
class PartialIndex:
    """Compact postings for one chunk of the corpus.

    Picklable on purpose: a process worker returns this, not the raw tokens.
    ``doc_id`` values are already the global ids the parent assigned.
    """

    postings: dict[str, Any]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    store_positions: bool
    representation: str


@dataclass(frozen=True)
class BuildReport:
    """One measured ``build_index_parallel`` call."""

    wall_seconds: float
    cpu_seconds: float
    peak_rss: int
    merge_seconds: float
    workers: int
    executor: str
    gil_enabled: bool | None
    documents: int


@dataclass(frozen=True)
class _IndexJob:
    """One chunk handed to ``build_partial``. Paths only — the worker reads."""

    paths: tuple[str, ...]
    start_doc_id: int
    positions: bool
    representation: str
    max_docs: int | None = None


def _is_buffer(representation: str) -> bool:
    """Postings stored as a pair of buffers, not a list of objects."""
    return representation in {"array", "numpy"}


def _check_representation(representation: str) -> None:
    if representation not in {"slots", "plain", "array", "numpy"}:
        raise ValueError(f"unknown representation: {representation!r}")


def _array_i_as_int32(buf: array) -> np.ndarray:
    """View an ``array('I')`` as ``int32`` without copying when it is safe."""
    if buf.itemsize != 4:
        return np.array(buf, dtype=np.int32)
    view = np.frombuffer(buf, dtype=np.uint32)
    if view.size and int(view.max()) >= 2**31:
        return view.astype(np.int32)
    return view.view(np.int32)


def _as_owned_int32(values: np.ndarray | array) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.dtype == np.int32 and values.flags.c_contiguous:
            return values
        return np.ascontiguousarray(values, dtype=np.int32)
    return _array_i_as_int32(values)


def _posting_arrays(
    index: Index, term: str
) -> tuple[np.ndarray, np.ndarray] | None:
    plist = index.postings.get(term)
    if not plist:
        return None
    if _is_buffer(index.representation):
        ids, tfs = plist
        return _as_owned_int32(ids), _as_owned_int32(tfs)
    count = len(plist)
    doc_ids = np.empty(count, dtype=np.int32)
    tfs = np.empty(count, dtype=np.int32)
    for i, posting in enumerate(plist):
        doc_ids[i] = posting.doc_id
        tfs[i] = posting.tf
    return doc_ids, tfs


def _accumulate_document(
    doc_id: int,
    doc: Document,
    *,
    positions: bool,
    representation: str,
    object_bucket: dict[str, list[Any]],
    array_ids: dict[str, array],
    array_tfs: dict[str, array],
    doc_lengths: dict[int, int],
    doc_meta: dict[int, DocMeta],
) -> None:
    n_tokens, stats = _term_stats(doc.text, positions=positions)
    doc_lengths[doc_id] = n_tokens
    stored_path = str(doc.path)
    if doc.source_line is not None:
        stored_path = f"{stored_path}#{doc.source_line}"
    doc_meta[doc_id] = DocMeta(path=stored_path, title=_title_from_document(doc))
    for term, (tf, pos) in stats.items():
        if _is_buffer(representation):
            array_ids[term].append(doc_id)
            array_tfs[term].append(tf)
        elif representation == "plain":
            object_bucket[term].append(PlainPosting(doc_id, tf, pos))
        else:
            object_bucket[term].append(Posting(doc_id, tf, pos))


def _partial_from_documents(
    docs: Iterable[Document],
    *,
    start_doc_id: int = 0,
    positions: bool = False,
    representation: str = "slots",
) -> PartialIndex:
    _check_representation(representation)
    object_bucket: dict[str, list[Any]] = defaultdict(list)
    array_ids: dict[str, array] = defaultdict(lambda: array("I"))
    array_tfs: dict[str, array] = defaultdict(lambda: array("I"))
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for offset, doc in enumerate(docs):
        _accumulate_document(
            start_doc_id + offset,
            doc,
            positions=positions,
            representation=representation,
            object_bucket=object_bucket,
            array_ids=array_ids,
            array_tfs=array_tfs,
            doc_lengths=doc_lengths,
            doc_meta=doc_meta,
        )
    return _freeze_partial(
        object_bucket,
        array_ids,
        array_tfs,
        doc_lengths,
        doc_meta,
        positions=positions,
        representation=representation,
    )


def _freeze_partial(
    object_bucket: dict[str, list[Any]],
    array_ids: dict[str, array],
    array_tfs: dict[str, array],
    doc_lengths: dict[int, int],
    doc_meta: dict[int, DocMeta],
    *,
    positions: bool,
    representation: str,
) -> PartialIndex:
    if _is_buffer(representation):
        postings: dict[str, Any] = {
            term: (array_ids[term], array_tfs[term]) for term in array_ids
        }
    else:
        postings = {}
        for term, rows in object_bucket.items():
            rows.sort(key=lambda item: item.doc_id)
            postings[term] = rows
    return PartialIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        store_positions=positions,
        representation=representation,
    )


def build_partial(
    doc_paths: list[Path] | Sequence[Path],
    start_doc_id: int = 0,
    *,
    positions: bool = False,
    representation: Representation = "slots",
    max_docs: int | None = None,
) -> PartialIndex:
    """Build postings for ``doc_paths``. Module-level so a process can call it.

    The worker reads each file itself. ``start_doc_id`` is the first global
    id of this chunk, assigned by the parent before any process starts, so
    ids never collide and a later merge can concatenate sorted lists.
    """
    docs: Iterable[Document] = iter_paths(doc_paths)
    if max_docs is not None:
        docs = islice(docs, max_docs)
    return _partial_from_documents(
        docs,
        start_doc_id=start_doc_id,
        positions=positions,
        representation=representation,
    )


def _build_partial_job(job: _IndexJob) -> PartialIndex:
    """Picklable adapter: ``ProcessPoolExecutor`` wants one argument."""
    return build_partial(
        [Path(path) for path in job.paths],
        job.start_doc_id,
        positions=job.positions,
        representation=job.representation,
        max_docs=job.max_docs,
    )


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Sorted-merge partial postings into one index.

    Chunks must carry disjoint ``doc_id``s. Object postings are merged with
    ``heapq.merge`` (each chunk's list is already sorted). Array postings are
    concatenated in chunk order, which is sorted when ids were assigned up front.
    """
    items = list(partials)
    if not items:
        return Index(
            postings={},
            doc_lengths={},
            doc_meta={},
            store_positions=False,
            representation="slots",
        )

    representations = {item.representation for item in items}
    if len(representations) != 1:
        raise ValueError(f"cannot merge mixed representations: {representations}")
    representation = items[0].representation
    positions = any(item.store_positions for item in items)

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    seen: set[int] = set()
    for item in items:
        overlap = seen.intersection(item.doc_lengths)
        if overlap:
            sample = min(overlap)
            raise ValueError(f"duplicate doc_id {sample} across partial indexes")
        seen.update(item.doc_lengths)
        doc_lengths.update(item.doc_lengths)
        doc_meta.update(item.doc_meta)

    if _is_buffer(representation):
        postings: dict[str, Any] = {}
        for item in items:
            for term, pair in item.postings.items():
                ids, tfs = pair
                slot = postings.get(term)
                if slot is None:
                    postings[term] = (array("I", ids), array("I", tfs))
                else:
                    slot[0].extend(ids)
                    slot[1].extend(tfs)
        if representation == "numpy":
            postings = {
                term: (
                    np.array(ids, dtype=np.int32),
                    np.array(tfs, dtype=np.int32),
                )
                for term, (ids, tfs) in postings.items()
            }
    else:
        grouped: dict[str, list[list[Any]]] = defaultdict(list)
        for item in items:
            for term, rows in item.postings.items():
                if rows:
                    grouped[term].append(rows)
        postings = {
            term: list(heap_merge(*lists, key=lambda posting: posting.doc_id))
            for term, lists in grouped.items()
        }

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        store_positions=positions,
        representation=representation,
    )


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
    The serial path ``merge([build_partial(all_paths)])`` builds the same
    index; this entry point stays for callers that already hold documents.
    """
    return merge(
        [
            _partial_from_documents(
                docs, positions=positions, representation=representation
            )
        ]
    )


def _split_counts(n_items: int, n_workers: int) -> list[int]:
    if n_items <= 0:
        return []
    n_workers = max(1, min(n_workers, n_items))
    base, extra = divmod(n_items, n_workers)
    return [base + (1 if i < extra else 0) for i in range(n_workers)]


def _plan_jobs(
    paths: Sequence[Path],
    *,
    workers: int,
    limit: int | None,
    positions: bool,
    representation: str,
) -> list[_IndexJob]:
    """Give each chunk a disjoint ``doc_id`` range before any worker starts.

    A file is never split across workers. ``.txt`` / ``.md`` contribute one
    id; a ``.jsonl`` contributes as many documents as it yields. ``limit``
    stops after that many documents (the first N files when each file is one
    document).
    """
    units: list[tuple[str, int, int | None]] = []
    remaining = limit
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            available = count_documents(path)
        elif suffix in TEXT_SUFFIXES:
            available = 1
        else:
            continue
        if available <= 0:
            continue
        take = available if remaining is None else min(available, remaining)
        max_docs = None if take == available else take
        units.append((str(path), take, max_docs))
        if remaining is not None:
            remaining -= take

    if not units:
        return []

    jobs: list[_IndexJob] = []
    offset = 0
    next_id = 0
    for count in _split_counts(len(units), workers):
        group = units[offset : offset + count]
        offset += count
        docs = sum(unit[1] for unit in group)
        partial = any(unit[2] is not None for unit in group)
        jobs.append(
            _IndexJob(
                paths=tuple(unit[0] for unit in group),
                start_doc_id=next_id,
                positions=positions,
                representation=representation,
                max_docs=docs if partial else None,
            )
        )
        next_id += docs
    return jobs


def _run_jobs(
    jobs: list[_IndexJob],
    executor: str,
    monitor: ResourceMonitor,
) -> list[PartialIndex]:
    if executor not in EXECUTORS:
        raise ValueError(
            f"unknown executor: {executor!r} (choose from {', '.join(EXECUTORS)})"
        )
    if not jobs or executor == "serial":
        return [_build_partial_job(job) for job in jobs]

    if executor == "threads":
        pool: ProcessPoolExecutor | ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=len(jobs)
        )
    else:
        # spawn re-imports the package in a fresh interpreter. Safe on Windows
        # and the only start method that stays correct with threads in-process.
        pool = ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=multiprocessing.get_context("spawn"),
        )

    futures = [pool.submit(_build_partial_job, job) for job in jobs]
    try:
        partials = [future.result() for future in futures]
    except Exception:
        # Don't wait for the other chunks: surface the worker error and stop.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    if executor == "processes":
        monitor.checkpoint()
    pool.shutdown(wait=True)
    return partials


def build_index_parallel(
    root: Path,
    *,
    positions: bool = False,
    representation: Representation = "slots",
    limit: int | None = None,
    workers: int | None = None,
    executor: ExecutorName = DEFAULT_EXECUTOR,
) -> tuple[Index, BuildReport]:
    """Index ``root`` with a serial loop, threads, or processes.

    Same ``build_partial`` and ``merge`` for every executor. ``workers`` is
    the number of chunks. The default is ``serial``: on this corpus the
    process-startup and pickle cost is larger than the time tokenization
    saves. Pass ``--executor processes`` when the corpus is big enough that
    the extra cores pay for that tax.
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be >= 1")
    if workers is None:
        workers = 1 if executor == "serial" else (os.cpu_count() or 1)

    paths = list_corpus_paths(root)
    jobs = _plan_jobs(
        paths,
        workers=workers,
        limit=limit,
        positions=positions,
        representation=representation,
    )
    launched = len(jobs) if jobs else 1

    monitor = ResourceMonitor()
    monitor.start()
    wall0 = time.perf_counter()
    try:
        partials = _run_jobs(jobs, executor, monitor)
        merge0 = time.perf_counter()
        index = merge(partials)
        merge_seconds = time.perf_counter() - merge0
        wall_seconds = time.perf_counter() - wall0
    finally:
        cpu_seconds, peak_rss = monitor.finish(children=executor == "processes")
    report = BuildReport(
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_rss=peak_rss,
        merge_seconds=merge_seconds,
        workers=launched,
        executor=executor,
        gil_enabled=gil_enabled(),
        documents=index.num_docs,
    )
    return index, report


def _format_report(index: Index, report: BuildReport) -> str:
    gil = report.gil_enabled
    gil_text = "n/a" if gil is None else str(gil)
    return "\n".join(
        [
            f"documents:      {index.n_docs()}",
            f"vocabulary:     {len(index.postings)}",
            f"tokens:         {sum(index.doc_lengths.values())}",
            f"representation: {index.representation}",
            f"positions:      {index.store_positions}",
            f"executor:       {report.executor}",
            f"workers:        {report.workers}",
            f"wall:           {report.wall_seconds:.3f} s",
            f"cpu:            {report.cpu_seconds:.3f} s",
            f"peak rss:       {format_bytes(report.peak_rss)} "
            f"({report.peak_rss} bytes)",
            f"merge:          {report.merge_seconds:.3f} s",
            f"gil_enabled:    {gil_text}",
        ]
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
        choices=("slots", "plain", "array", "numpy"),
        default="slots",
        help="postings storage used for the Lab 2 memory table",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="how many chunks to build (default: 1 for serial, cpu_count otherwise)",
    )
    parser.add_argument(
        "--executor",
        choices=EXECUTORS,
        default=DEFAULT_EXECUTOR,
        help="serial loop, threads, or processes (default: serial)",
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

    try:
        index, report = build_index_parallel(
            args.root,
            positions=args.positions,
            representation=args.representation,
            limit=args.limit,
            workers=args.workers,
            executor=args.executor,
        )
    except Exception:
        # Worker failures arrive with the original traceback attached as
        # ``__cause__`` (``_RemoteTraceback`` under spawn). Print both.
        traceback.print_exc()
        return 1

    from findex.store import save

    t1 = time.perf_counter()
    save(index, args.out)
    save_elapsed = time.perf_counter() - t1
    size = args.out.stat().st_size

    print(_format_report(index, report))
    print(f"save elapsed:   {save_elapsed:.3f} s")
    print(f"file size:      {format_bytes(size)} ({size} bytes)")
    print(f"wrote:          {args.out}")
    return 0


if __name__ == "__main__":
    # spawn re-imports this module as __mp_main__, so the pool is not started
    # again here. freeze_support covers the frozen-executable case on Windows.
    multiprocessing.freeze_support()
    # Re-import so pickle records Posting/Index as findex.index.*, not __main__.
    from findex.index import main as _main

    sys.exit(_main())

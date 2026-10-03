"""Persist an inverted index and load it back.

Two formats, chosen by the path suffix:

* ``.pkl`` / ``.pickle`` / ``.bin`` — pickle (fast, compact, trusted files only)
* ``.json`` — JSON (safe, portable, slower and larger)

Never ``pickle.load`` a file you did not write. The pickle opcode stream is
a program: a crafted file can execute arbitrary Python on load (remote code
execution). That is fine for *this* machine's own ``index.bin``; it is not
fine for anything downloaded, emailed, or checked in by a stranger.
"""

from __future__ import annotations

import json
import pickle
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from findex.index import DocMeta, Index, PlainPosting, Posting
from findex.timing import timed

PICKLE_SUFFIXES = {".pkl", ".pickle", ".bin"}
JSON_FORMAT = "findex-json-v1"


@contextmanager
def open_index(path: Path | str) -> Iterator[Index]:
    """Load an index and always ``close()`` it, including on exceptions."""
    index = load(path)
    try:
        yield index
    finally:
        index.close()


def save(index: Index, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        _save_json(index, path)
        return
    if suffix in PICKLE_SUFFIXES or suffix == "":
        _save_pickle(index, path)
        return
    raise ValueError(f"unknown index suffix {suffix!r}; use .pkl/.bin or .json")


@timed
def load(path: Path | str) -> Index:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix in PICKLE_SUFFIXES or suffix == "":
        return _load_pickle(path)
    # Sniff: JSON objects start with '{'.
    head = path.read_bytes()[:1]
    if head == b"{":
        return _load_json(path)
    return _load_pickle(path)


def _save_pickle(index: Index, path: Path) -> None:
    # pickle.dumps of our own Index is convenient. pickle.loads of anyone
    # else's file is a shell. See the module docstring.
    with path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Index:
    # SECURITY: pickle.load executes bytecode from the file. Only call this
    # on an index produced by findex on a trusted path — never on a file
    # from the network, email, or an untrusted checkout.
    with path.open("rb") as fh:
        obj = pickle.load(fh)  # noqa: S301 — trusted local index only
    if not isinstance(obj, Index):
        raise TypeError(f"{path} did not contain an Index")
    return obj


def _posting_row(posting: Posting | PlainPosting) -> list:
    row: list = [posting.doc_id, posting.tf]
    if posting.positions:
        row.append(list(posting.positions))
    return row


def _save_json(index: Index, path: Path) -> None:
    postings_json: dict[str, list] = {}
    for term, plist in index.postings.items():
        if index.representation == "array":
            ids, tfs = plist
            postings_json[term] = [list(ids), list(tfs)]
        else:
            postings_json[term] = [_posting_row(p) for p in plist]

    payload = {
        "format": JSON_FORMAT,
        "representation": index.representation,
        "positions": index.store_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): {"path": m.path, "title": m.title}
            for k, m in index.doc_meta.items()
        },
        "postings": postings_json,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Index:
    payload = json.loads(path.read_text(encoding="utf-8"))
    representation = payload.get("representation", "slots")
    store_positions = bool(payload.get("positions", False))
    doc_lengths = {int(k): int(v) for k, v in payload["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(path=v["path"], title=v["title"])
        for k, v in payload["doc_meta"].items()
    }
    postings: dict = {}
    for term, raw_data in payload["postings"].items():
        if representation == "array":
            ids_raw, tfs_raw = raw_data
            postings[term] = (
                array("I", (int(x) for x in ids_raw)),
                array("I", (int(x) for x in tfs_raw)),
            )
            continue
            
        rebuilt = []
        for row in raw_data:
            doc_id, tf = int(row[0]), int(row[1])
            positions = tuple(int(p) for p in row[2]) if len(row) > 2 else ()
            if representation == "plain":
                rebuilt.append(PlainPosting(doc_id, tf, positions))
            else:
                rebuilt.append(Posting(doc_id, tf, positions))
        postings[term] = rebuilt
        
    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        store_positions=store_positions,
        representation=representation,
    )

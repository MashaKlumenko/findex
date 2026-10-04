"""Partial indexes, merge, and the three executors produce one index."""

from __future__ import annotations

import pickle
import traceback
from pathlib import Path

import pytest

from findex.corpus import iter_documents, list_corpus_paths
from findex.index import (
    build_index,
    build_index_parallel,
    build_partial,
    merge,
)
from findex.resources import ResourceMonitor, gil_enabled


def _tiny_corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("red cat sat", encoding="utf-8")
    (tmp_path / "b.txt").write_text("blue cat ran", encoding="utf-8")
    (tmp_path / "c.txt").write_text("red dog sat", encoding="utf-8")
    return tmp_path


def _dump(index: object) -> bytes:
    return pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)


def test_serial_merge_matches_lab4(tmp_path: Path) -> None:
    """merge([build_partial(all paths)]) is the Lab 4 index, byte for byte."""
    root = _tiny_corpus(tmp_path)
    paths = list_corpus_paths(root)
    lab4 = build_index(iter_documents(root), positions=True)
    serial = merge([build_partial(paths, positions=True)])
    assert serial.n_docs() == lab4.n_docs()
    assert serial.doc_lengths == lab4.doc_lengths
    assert _dump(serial) == _dump(lab4)


def test_executors_match_on_clean_corpus(tmp_path: Path) -> None:
    root = _tiny_corpus(tmp_path)
    lab4 = build_index(iter_documents(root))
    expected = _dump(lab4)
    for executor in ("serial", "threads", "processes"):
        index, report = build_index_parallel(
            root, executor=executor, workers=2  # type: ignore[arg-type]
        )
        assert report.executor == executor
        assert _dump(index) == expected
        assert set(index.doc_lengths) == {0, 1, 2}


def test_jsonl_ranges_stay_unique(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("red cat", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(
        '{"text": "blue cat sat"}\n{"text": "green dog"}\n',
        encoding="utf-8",
    )
    (tmp_path / "c.txt").write_text("red dog sat", encoding="utf-8")
    lab4 = build_index(iter_documents(tmp_path), positions=True)
    index, _report = build_index_parallel(
        tmp_path, executor="threads", workers=3, positions=True
    )
    assert _dump(index) == _dump(lab4)
    assert sorted(index.doc_lengths) == [0, 1, 2, 3]


def test_merge_rejects_duplicate_doc_ids(tmp_path: Path) -> None:
    root = _tiny_corpus(tmp_path)
    paths = list_corpus_paths(root)
    first = build_partial(paths)
    second = build_partial(paths)
    with pytest.raises(ValueError, match="duplicate doc_id"):
        merge([first, second])


def test_build_partial_is_importable_at_module_level() -> None:
    import findex.index as index_mod

    assert index_mod.build_partial.__module__ == "findex.index"
    # Process pools pickle the function by its module path, not by value.
    pickle.loads(pickle.dumps(index_mod._build_partial_job))


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_worker_failure_surfaces(tmp_path: Path, executor: str) -> None:
    root = _tiny_corpus(tmp_path)
    with pytest.raises(ValueError, match="unknown representation") as caught:
        build_index_parallel(
            root,
            executor=executor,  # type: ignore[arg-type]
            workers=2,
            representation="bogus",
        )
    rendered = "".join(traceback.format_exception(caught.value))
    cause = caught.value.__cause__
    if cause is not None:
        rendered += str(cause)
    assert "build_partial" in rendered or "bogus" in rendered


def test_array_and_positions_round_trip_processes(tmp_path: Path) -> None:
    root = _tiny_corpus(tmp_path)
    lab4 = build_index(iter_documents(root), representation="array")
    index, _report = build_index_parallel(
        root, executor="processes", workers=2, representation="array"
    )
    assert index.representation == "array"
    assert index.doc_ids_for("cat") == lab4.doc_ids_for("cat")
    assert _dump(index) == _dump(lab4)


def test_resource_monitor_reports_rss() -> None:
    monitor = ResourceMonitor()
    monitor.start()
    total = 0
    for i in range(100_000):
        total += i
    cpu, rss = monitor.finish(children=False)
    assert rss > 0
    assert cpu >= 0
    assert total > 0
    assert gil_enabled() is None or isinstance(gil_enabled(), bool)

"""Inverted index construction, Boolean merge, and save/load."""

from __future__ import annotations

from pathlib import Path

from findex.corpus import iter_documents
from findex.index import DocMeta, Posting, build_index
from findex.search import merge_and, merge_not, merge_or, search
from findex.store import load, open_index, save


def _tiny_corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("red cat sat", encoding="utf-8")
    (tmp_path / "b.txt").write_text("blue cat ran", encoding="utf-8")
    (tmp_path / "c.txt").write_text("red dog sat", encoding="utf-8")
    return tmp_path


def test_posting_and_docmeta_are_frozen_slotted_and_hashable() -> None:
    posting = Posting(1, 2, (0, 3))
    meta = DocMeta("a.txt", "red cat sat")
    assert Posting.__slots__ == ("doc_id", "tf", "positions")
    assert DocMeta.__slots__ == ("path", "title")
    assert {posting, meta} == {Posting(1, 2, (0, 3)), DocMeta("a.txt", "red cat sat")}
    try:
        posting.doc_id = 9  # type: ignore[misc]
        raise AssertionError("Posting should be frozen")
    except AttributeError:
        pass


def test_build_is_one_pass_sorted_postings(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)))
    assert index.n_docs() == 3
    assert index.doc_lengths[0] == 3
    cat = index.postings["cat"]
    assert [p.doc_id for p in cat] == [0, 1]
    assert all(a.doc_id <= b.doc_id for a, b in zip(cat, cat[1:], strict=False))
    assert cat[0].tf == 1
    assert index.doc_meta[0].path.endswith("a.txt")
    assert "red" in index.doc_meta[0].title


def test_positions_flag(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)), positions=True)
    sat = {p.doc_id: p for p in index.postings["sat"]}
    assert sat[0].positions == (2,)
    assert sat[2].positions == (2,)


def test_array_representation(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)), representation="array")
    assert index.representation == "array"
    assert list(index.postings["cat"][0]) == [0, 1]
    assert index.doc_ids_for("cat") == [0, 1]


def test_merge_algorithms() -> None:
    a = [1, 3, 5, 7]
    b = [3, 4, 7, 9]
    assert merge_and(a, b) == [3, 7]
    assert merge_or(a, b) == [1, 3, 4, 5, 7, 9]
    assert merge_not(a, b) == [1, 5]


def test_boolean_search_merge_and_set(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)))
    for engine in ("merge", "set"):
        assert search(index, "cat", engine=engine) == [0, 1]
        assert search(index, "red cat", engine=engine) == [0]
        assert search(index, "red OR blue", engine=engine) == [0, 1, 2]
        assert search(index, "sat NOT dog", engine=engine) == [0]
        assert search(index, "NOT cat", engine=engine) == [2]
        assert search(index, "unicorn", engine=engine) == []
        assert search(index, "", engine=engine) == []


def test_pickle_and_json_roundtrip(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)), positions=True)
    pkl = tmp_path / "index.pkl"
    js = tmp_path / "index.json"
    save(index, pkl)
    save(index, js)
    for path in (pkl, js):
        loaded = load(path)
        assert loaded.n_docs() == 3
        assert loaded.doc_ids_for("cat") == [0, 1]
        assert loaded.postings["sat"][0].positions == (2,)
        assert search(loaded, "red cat") == [0]


def test_index_mapping_protocol(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)))
    assert len(index) == len(index.postings)
    assert "cat" in index
    assert "unicorn" not in index
    assert index["cat"][0].doc_id == 0
    assert set(index) == set(index.postings)
    assert "Index(terms=" in repr(index)
    assert "docs=3" in repr(index)
    assert index.num_docs == 3
    assert index.avg_doc_length == 3.0
    assert index.doc_length(0) == 3
    assert index.df("cat") == 2
    assert index.df("unicorn") == 0
    try:
        index["unicorn"]
        raise AssertionError("missing term should KeyError")
    except KeyError:
        pass


def test_open_index_closes_on_exception(tmp_path: Path) -> None:
    index = build_index(iter_documents(_tiny_corpus(tmp_path)))
    path = tmp_path / "ix.pkl"
    save(index, path)
    try:
        with open_index(path) as ix:
            assert "cat" in ix
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert ix._closed is True
    assert len(ix.postings) == 0


def test_timed_preserves_names() -> None:
    from findex.index import build_index
    from findex.rank import ranked_search
    from findex.store import load

    assert build_index.__name__ == "build_index"
    assert load.__name__ == "load"
    assert ranked_search.__name__ == "ranked_search"


def test_cli_index_and_search(tmp_path: Path, capsys) -> None:
    from findex.index import main as index_main
    from findex.search import main as search_main

    root = _tiny_corpus(tmp_path)
    out = tmp_path / "index.bin"
    assert index_main([str(root), "--out", str(out)]) == 0
    captured = capsys.readouterr().out
    assert "documents:" in captured
    assert search_main([str(out), "red cat", "--boolean", "--engine", "merge"]) == 0
    out_text = capsys.readouterr().out
    assert "hits:" in out_text
    assert "red cat sat" in out_text

"""Lab 3 ranking checks, cache traces, and precision@5.

    python scripts/lab03_eval.py data/ --index data/index.pkl
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.corpus import iter_documents  # noqa: E402
from findex.index import build_index  # noqa: E402
from findex.query import match_doc_ids  # noqa: E402
from findex.rank import BM25, TfIdf, ranked_search  # noqa: E402
from findex.store import open_index, save  # noqa: E402

# Hand labels: a result is relevant if its path contains the book slug.
LABELED_QUERIES: list[tuple[str, list[str]]] = [
    ("elizabeth darcy", ["pride-and-prejudice"]),
    ("white rabbit", ["alice-in-wonderland"]),
    ("sherlock holmes", ["sherlock-holmes"]),
    ('"dorian gray"', ["dorian-gray"]),
    ("frankenstein creature", ["frankenstein"]),
    ("huckleberry finn", ["huckleberry-finn"]),
    ('"best of times"', ["tale-of-two-cities"]),
    ("jonathan harker", ["dracula"]),
    ("baskerville", ["sherlock-holmes"]),
    ("mad hatter", ["alice-in-wonderland"]),
]


def precision_at_k_index(index, results, markers: list[str], k: int = 5) -> float:
    top = results[:k]
    if not top:
        return 0.0
    n = 0
    for row in top:
        meta = index.doc_meta[row.doc_id]
        blob = f"{meta.path} {meta.title}".replace("\\", "/").lower()
        if any(m.lower() in blob for m in markers):
            n += 1
    return n / k


def sanity_synthetic(tmp: Path) -> list[str]:
    lines = ["## Synthetic ranking checks (same Index / Scorer as the corpus)"]
    (tmp / "rare.txt").write_text("xyzzy appears once here\n", encoding="utf-8")
    (tmp / "common.txt").write_text(
        "the the the the the filler words about nothing\n", encoding="utf-8"
    )
    pad = " ".join(["lorem"] * 30)
    (tmp / "tf01.txt").write_text("needle " + pad + "\n", encoding="utf-8")
    (tmp / "tf19.txt").write_text(" ".join(["needle"] * 19) + " " + pad + "\n", encoding="utf-8")
    (tmp / "tf20.txt").write_text(" ".join(["needle"] * 20) + " " + pad + "\n", encoding="utf-8")
    (tmp / "short.txt").write_text("walrus\n", encoding="utf-8")
    (tmp / "long.txt").write_text(
        "walrus " + " ".join(["padding"] * 400) + "\n", encoding="utf-8"
    )
    index = build_index(iter_documents(tmp))

    rare = ranked_search(index, "xyzzy the", scorer=TfIdf(), k=2, snippets=False)
    rare_stems = [Path(index.doc_meta[r.doc_id].path).stem for r in rare]
    lines.append(
        f"1. rare vs common: {rare_stems[0]!r} score={rare[0].score:.3f} "
        f"> {rare_stems[1]!r} score={rare[1].score:.3f}"
    )

    bm25 = BM25()
    by = {
        Path(index.doc_meta[r.doc_id].path).stem: r.score
        for r in ranked_search(index, "needle", scorer=bm25, k=3, snippets=False)
    }
    early = by["tf19"] - by["tf01"]
    late = by["tf20"] - by["tf19"]
    lines.append(
        f"2. BM25 saturation: Δ(tf19-tf01)={early:.4f}  Δ(tf20-tf19)={late:.4f} "
        f"(late/early={late / early:.3f})"
    )

    length = ranked_search(index, "walrus", scorer=bm25, k=2, snippets=False)
    stems = [Path(index.doc_meta[r.doc_id].path).stem for r in length]
    lines.append(
        f"3. short vs long: {stems[0]} {length[0].score:.3f} > "
        f"{stems[1]} {length[1].score:.3f}"
    )
    return lines


def eval_corpus(index, k: int = 5) -> list[str]:
    lines = [
        f"| Query | TF-IDF P@{k} | BM25 P@{k} |",
        "|---|---:|---:|",
    ]
    tf_scores: list[float] = []
    bm_scores: list[float] = []
    for query, markers in LABELED_QUERIES:
        tf = ranked_search(index, query, scorer=TfIdf(), k=k, snippets=False)
        bm = ranked_search(index, query, scorer=BM25(), k=k, snippets=False)
        p_tf = precision_at_k_index(index, tf, markers, k)
        p_bm = precision_at_k_index(index, bm, markers, k)
        tf_scores.append(p_tf)
        bm_scores.append(p_bm)
        lines.append(f"| `{query}` | {p_tf:.2f} | {p_bm:.2f} |")
    lines.append(
        f"| **macro average** | "
        f"{sum(tf_scores) / len(tf_scores):.2f} | "
        f"{sum(bm_scores) / len(bm_scores):.2f} |"
    )
    return lines


def cache_trace(index) -> list[str]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    q = "elizabeth darcy" if "elizabeth" in index else next(iter(index))
    match_doc_ids(index, q)
    match_doc_ids(index, q)
    return ["repeated query ran; see INFO logs for @timed ms and lru_cache hit"]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=ROOT / "data")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        for line in sanity_synthetic(Path(tmp)):
            print(line)

    print()
    corpus = args.root
    if not corpus.exists():
        print(f"no corpus at {corpus}; skip P@5")
        return 0

    if args.index and args.index.exists() and not args.rebuild:
        with open_index(args.index) as index:
            print(index)
            for line in eval_corpus(index):
                print(line)
            print()
            cache_trace(index)
        return 0

    print("building index with positions...")
    index = build_index(iter_documents(corpus), positions=True)
    print(index)
    if args.index:
        save(index, args.index)
        print(f"wrote {args.index}")
    for line in eval_corpus(index):
        print(line)
    print()
    cache_trace(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

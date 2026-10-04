"""Precision@5 for keyword, semantic, and hybrid search.

The 10 queries are the Lab 3 labels: a hit counts when the passage belongs
to the labeled book. The extra five are paraphrases that avoid the names
and phrases those books are retrieved by.

    python scripts/lab08_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.rank import BM25, ranked_search  # noqa: E402
from findex.semantic import (  # noqa: E402
    Embeddings,
    embed_index,
    hybrid_search,
    load_encoder,
    semantic_search,
)
from findex.store import load  # noqa: E402

LABELED: list[tuple[str, list[str]]] = [
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

PARAPHRASES: list[tuple[str, list[str]]] = [
    (
        "a proud young woman and a reserved gentleman",
        ["pride-and-prejudice"],
    ),
    (
        "a child tumbles into a strange underground tea party",
        ["alice-in-wonderland"],
    ),
    (
        "a consulting detective shares rooms with a doctor",
        ["sherlock-holmes"],
    ),
    (
        "a student assembles a living body and abandons it",
        ["frankenstein"],
    ),
    (
        "a painted portrait ages while the sitter stays young",
        ["dorian-gray"],
    ),
]


def precision(index, hits, markers: list[str], k: int = 5) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    good = 0
    for hit in top:
        meta = index.doc_meta[hit.doc_id]
        blob = f"{meta.path} {meta.title}".replace("\\", "/").lower()
        if any(marker in blob for marker in markers):
            good += 1
    return good / k


def main() -> None:
    index = load(ROOT / "data" / "index.pkl")
    emb_dir = ROOT / "data" / "embeddings"
    encode = load_encoder("all-MiniLM-L6-v2", progress=not emb_dir.joinpath("vectors.npy").is_file())
    if emb_dir.joinpath("vectors.npy").is_file():
        embeddings = Embeddings.load(emb_dir)
        print(f"loaded {embeddings.vectors.shape} from {emb_dir}")
    else:
        print("embedding corpus...")
        embeddings = embed_index(index, encode, model="all-MiniLM-L6-v2")
        embeddings.save(emb_dir)
        print(f"wrote {embeddings.vectors.shape} to {emb_dir}")

    rows = [("lexical", query, markers) for query, markers in LABELED]
    rows += [("paraphrase", query, markers) for query, markers in PARAPHRASES]
    print("| kind | query | keyword | semantic | hybrid |")
    print("|---|---|---:|---:|---:|")
    totals = {"keyword": [], "semantic": [], "hybrid": []}
    for kind, query, markers in rows:
        keyword = ranked_search(
            index, query, scorer=BM25(), k=5, snippets=False, engine="numpy"
        )
        semantic = semantic_search(
            index, embeddings, query, encode, k=5, snippets=False
        )
        hybrid = hybrid_search(
            index, embeddings, query, encode, k=5, snippets=False
        )
        scores = {
            "keyword": precision(index, keyword, markers),
            "semantic": precision(index, semantic, markers),
            "hybrid": precision(index, hybrid, markers),
        }
        for name, value in scores.items():
            totals[name].append(value)
        shown = query.replace("|", "\\|")
        print(
            f"| {kind} | `{shown}` | {scores['keyword']:.2f} | "
            f"{scores['semantic']:.2f} | {scores['hybrid']:.2f} |"
        )
    n = len(rows)
    print(
        "| | **macro average** | "
        f"{sum(totals['keyword']) / n:.2f} | "
        f"{sum(totals['semantic']) / n:.2f} | "
        f"{sum(totals['hybrid']) / n:.2f} |"
    )


if __name__ == "__main__":
    main()

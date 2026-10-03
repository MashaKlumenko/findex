"""Lab 2 memory table: three postings representations on the same corpus.

    python scripts/memory_study.py data/
"""

from __future__ import annotations

import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.corpus import iter_documents  # noqa: E402
from findex.index import build_index  # noqa: E402
from findex.stats import format_bytes  # noqa: E402
from findex.store import load, save  # noqa: E402


def measure(root: Path, representation: str) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(iter_documents(root), representation=representation)
    build_elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.pkl"
        t1 = time.perf_counter()
        save(index, path)
        save_elapsed = time.perf_counter() - t1
        size = path.stat().st_size
        t2 = time.perf_counter()
        loaded = load(path)
        load_elapsed = time.perf_counter() - t2
    del loaded
    return {
        "representation": representation,
        "docs": index.n_docs(),
        "vocab": len(index.postings),
        "peak": peak,
        "build_s": build_elapsed,
        "save_s": save_elapsed,
        "load_s": load_elapsed,
        "size": size,
    }


def measure_formats(root: Path) -> list[dict]:
    index = build_index(iter_documents(root), representation="slots")
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, suffix in (("pickle", ".pkl"), ("json", ".json")):
            path = Path(tmp) / f"index{suffix}"
            t0 = time.perf_counter()
            save(index, path)
            save_s = time.perf_counter() - t0
            size = path.stat().st_size
            t1 = time.perf_counter()
            load(path)
            load_s = time.perf_counter() - t1
            rows.append(
                {"name": name, "size": size, "save_s": save_s, "load_s": load_s}
            )
    pos = build_index(iter_documents(root), representation="slots", positions=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.pkl"
        save(pos, path)
        rows.append(
            {
                "name": "pickle+positions",
                "size": path.stat().st_size,
                "save_s": 0.0,
                "load_s": 0.0,
            }
        )
    return rows


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    print(f"corpus: {root}")
    rows = [measure(root, name) for name in ("plain", "slots", "array")]
    print()
    print(
        f"{'Postings representation':<42} "
        f"{'Peak memory (build)':>20} {'Index file size':>16} {'Load time':>10}"
    )
    # Оновлена мітка, що відповідає лінійній структурі пам'яті
    labels = {
        "plain": "list[Posting] with plain @dataclass",
        "slots": "list[Posting] with slots=True",
        "array": "array('I') contiguous buffers — no objects",
    }
    for row in rows:
        print(
            f"{labels[row['representation']]:<42} "
            f"{format_bytes(row['peak']):>20} "
            f"{format_bytes(row['size']):>16} "
            f"{row['load_s']:9.3f}s"
        )
    print()
    for row in rows:
        print(
            f"{row['representation']}: docs={row['docs']} vocab={row['vocab']} "
            f"build={row['build_s']:.3f}s save={row['save_s']:.3f}s "
            f"peak_bytes={row['peak']} size_bytes={row['size']}"
        )
    print()
    print("Persistence (slots postings, same corpus):")
    for fmt in measure_formats(root):
        print(
            f"  {fmt['name']:<18} size={format_bytes(fmt['size']):>10} "
            f"save={fmt['save_s']:.3f}s load={fmt['load_s']:.3f}s "
            f"({fmt['size']} bytes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

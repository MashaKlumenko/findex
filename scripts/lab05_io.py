"""I/O-bound counterexample: threads scale when the GIL is released.

Two workloads:

* reading every corpus file (often RAM-cache + decode, so it may stay
  CPU-ish on a warm disk);
* ``time.sleep``, which drops the GIL the same way a blocking ``read`` does
  when the disk or the network is actually waiting.

    python scripts/lab05_io.py --root data
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.corpus import list_corpus_paths  # noqa: E402
from findex.resources import gil_enabled  # noqa: E402


def _chunks(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    base, extra = divmod(len(items), n)
    out = []
    offset = 0
    for i in range(n):
        count = base + (1 if i < extra else 0)
        out.append(items[offset : offset + count])
        offset += count
    return out


def _read_chunk(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        total += len(path.read_bytes())
    return total


def _sleep_chunk(delays: list[float]) -> float:
    total = 0.0
    for delay in delays:
        time.sleep(delay)
        total += delay
    return total


def _time_threads(fn, pieces: list, workers: int) -> float:
    if workers == 1:
        t0 = time.perf_counter()
        for piece in pieces:
            fn(piece)
        return time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fn, pieces))
    return time.perf_counter() - t0


def _median_run(fn, work: list, workers: int, runs: int) -> float:
    samples = [_time_threads(fn, _chunks(work, workers), workers) for _ in range(runs)]
    return float(statistics.median(samples))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args(argv)

    logical = os.cpu_count() or 1
    counts = []
    for n in (1, 2, 4, 8, logical):
        if n not in counts:
            counts.append(n)

    paths = list_corpus_paths(args.root)
    # Touch the files once so the timed runs are not a cold-cache accident.
    _read_chunk(paths)

    print(f"python {sys.version.split()[0]}  sys._is_gil_enabled() = {gil_enabled()}")
    print(f"files: {len(paths)}")
    print("read_bytes (warm cache), median wall seconds")
    read_baseline = None
    for n in counts:
        wall = _median_run(_read_chunk, paths, n, args.runs)
        if read_baseline is None:
            read_baseline = wall
        speedup = read_baseline / wall if wall else 0
        print(f"  threads={n:<3}  wall={wall:.3f}s  speedup={speedup:.2f}x")

    delays = [0.05] * 40
    print("time.sleep(0.05) x 40  (blocking wait, GIL released)")
    sleep_baseline = None
    for n in counts:
        wall = _median_run(_sleep_chunk, delays, n, args.runs)
        if sleep_baseline is None:
            sleep_baseline = wall
        speedup = sleep_baseline / wall if wall else 0
        print(f"  threads={n:<3}  wall={wall:.3f}s  speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
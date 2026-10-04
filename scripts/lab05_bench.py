"""Median-of-3 timings for serial, threaded, and process index builds.

Each sample is a new interpreter so the parent's peak RSS starts from zero.
Wall time is ``perf_counter`` around ``build_partial`` + ``merge``. CPU time
includes worker processes. RSS is the parent working set.

    python scripts/lab05_bench.py --root data --out docs/lab05-bench.json
    python scripts/lab05_bench.py --once --executor threads --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.index import build_index_parallel  # noqa: E402
from findex.resources import gil_enabled  # noqa: E402


def cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            name, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(name).strip()
    return platform.processor() or platform.machine()


def worker_counts(explicit: list[int] | None) -> list[int]:
    if explicit:
        return explicit
    logical = os.cpu_count() or 1
    counts: list[int] = []
    for n in (1, 2, 4, 8, logical):
        if n >= 1 and n not in counts:
            counts.append(n)
    return counts


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def run_once(root: Path, executor: str, workers: int) -> None:
    _index, report = build_index_parallel(
        root, executor=executor, workers=workers  # type: ignore[arg-type]
    )
    payload = {
        "executor": report.executor,
        "workers": report.workers,
        "wall_seconds": report.wall_seconds,
        "cpu_seconds": report.cpu_seconds,
        "peak_rss": report.peak_rss,
        "merge_seconds": report.merge_seconds,
        "documents": report.documents,
        "gil_enabled": report.gil_enabled,
        "python": platform.python_version(),
    }
    sys.stdout.write(json.dumps(payload) + "\n")


def _sample(root: Path, executor: str, workers: int) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--once",
        "--root",
        str(root),
        "--executor",
        executor,
        "--workers",
        str(workers),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"{executor} workers={workers} failed with code {proc.returncode}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def run_grid(args: argparse.Namespace) -> None:
    root = Path(args.root)
    counts = worker_counts(args.workers)
    executors = args.executor or ["serial", "threads", "processes"]
    cells: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gil_enabled": gil_enabled(),
        "runs": args.runs,
        "root": str(root),
    }
    print(f"machine: {meta['cpu_model']}  logical={meta['logical_cpus']}", flush=True)
    print(f"python {meta['python']}  gil_enabled={meta['gil_enabled']}", flush=True)

    if not args.no_warmup:
        print("warmup (discarded)", flush=True)
        _sample(root, "serial", 1)

    for executor in executors:
        use_counts = [1] if executor == "serial" and not args.serial_all else counts
        if executor == "serial" and not args.serial_all:
            use_counts = [1]
        for workers in use_counts:
            samples = []
            for i in range(args.runs):
                t0 = time.perf_counter()
                sample = _sample(root, executor, workers)
                samples.append(sample)
                print(
                    f"  {executor:10} n={workers:<3} "
                    f"run {i + 1}/{args.runs}  "
                    f"wall={sample['wall_seconds']:.3f}s  "
                    f"cpu={sample['cpu_seconds']:.3f}s  "
                    f"rss={sample['peak_rss'] / (1024 * 1024):.1f} MiB  "
                    f"merge={sample['merge_seconds']:.3f}s  "
                    f"(outer {time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
            walls = [s["wall_seconds"] for s in samples]
            cell = {
                "executor": executor,
                "workers": workers,
                "python": samples[-1]["python"],
                "gil_enabled": samples[-1]["gil_enabled"],
                "documents": samples[-1]["documents"],
                "wall_median": median(walls),
                "wall_min": min(walls),
                "wall_max": max(walls),
                "cpu_median": median([s["cpu_seconds"] for s in samples]),
                "rss_median": median([float(s["peak_rss"]) for s in samples]),
                "merge_median": median([s["merge_seconds"] for s in samples]),
                "samples": samples,
            }
            cells.append(cell)
            _write(out_path, meta, cells)
            print(
                f"  -> median wall={cell['wall_median']:.3f}s "
                f"spread={cell['wall_min']:.3f}..{cell['wall_max']:.3f}",
                flush=True,
            )

    _write(out_path, meta, cells)
    print(f"wrote {out_path}", flush=True)


def _write(path: Path, meta: dict, cells: list[dict]) -> None:
    serial = next(
        (
            c["wall_median"]
            for c in cells
            if c["executor"] == "serial" and c["workers"] == 1
        ),
        None,
    )
    # A free-threaded threads-only run has no serial row. Scale against n=1.
    baseline = serial
    if baseline is None:
        baseline = next(
            (
                c["wall_median"]
                for c in cells
                if c["executor"] == "threads" and c["workers"] == 1
            ),
            None,
        )
    enriched = []
    for cell in cells:
        row = dict(cell)
        if baseline:
            row["speedup"] = baseline / cell["wall_median"]
        enriched.append(row)
    path.write_text(
        json.dumps({"meta": meta, "cells": enriched}, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "lab05-bench.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--executor", action="append", default=None)
    parser.add_argument("--workers", type=int, action="append", default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--serial-all",
        action="store_true",
        help="also run the serial executor at every worker count",
    )
    args = parser.parse_args(argv)
    if args.once:
        executor = (args.executor or ["serial"])[0]
        workers = (args.workers or [1])[0]
        run_once(args.root, executor, workers)
        return 0
    run_grid(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

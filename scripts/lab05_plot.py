"""Speedup-vs-workers plot from one or more lab05 benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _label(cell: dict, meta: dict) -> str:
    gil = cell.get("gil_enabled")
    if gil is False:
        return "threads, 3.13t (GIL off)"
    name = cell["executor"]
    if name == "threads":
        return "threads (GIL on)"
    if name == "processes":
        return "processes"
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "lab05-speedup.png",
    )
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    loaded = []
    cpu_model = ""
    logical = ""
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        cpu_model = cpu_model or str(meta.get("cpu_model", ""))
        logical = logical or str(meta.get("logical_cpus", ""))
        loaded.append((meta, payload["cells"]))

    baseline = None
    for _meta, cells in loaded:
        for cell in cells:
            if cell["executor"] == "serial" and int(cell["workers"]) == 1:
                baseline = float(cell["wall_median"])
    if baseline is None:
        for _meta, cells in loaded:
            for cell in cells:
                if int(cell["workers"]) == 1:
                    baseline = float(cell["wall_median"])
                    break
            if baseline is not None:
                break

    series: dict[str, list[tuple[int, float]]] = {}
    for meta, cells in loaded:
        for cell in cells:
            if cell["executor"] == "serial":
                continue
            wall = float(cell["wall_median"])
            speedup = (baseline / wall) if baseline else float(cell.get("speedup", 1))
            series.setdefault(_label(cell, meta), []).append(
                (int(cell["workers"]), speedup)
            )

    if not series:
        print("no cells to plot", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    max_workers = 1
    for label, points in series.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        max_workers = max(max_workers, max(xs))
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    ideal_x = [1, max_workers]
    ax.plot(
        ideal_x,
        ideal_x,
        linestyle="--",
        color="0.45",
        linewidth=1.5,
        label="ideal linear",
    )
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup  (serial wall / this wall)")
    # The ideal line is y = workers (20× at 20 workers). Clip the axis so the
    # measured curves stay readable; the dashed line leaving the frame is the
    # gap Amdahl and startup cost open up.
    ax.set_ylim(0, 4.2)
    title = "Index build speedup"
    if cpu_model:
        title += f"\n{cpu_model}"
        if logical:
            title += f"  ·  {logical} logical CPUs"
    ax.set_title(title)
    ax.set_xticks(sorted({x for points in series.values() for x, _y in points}))
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

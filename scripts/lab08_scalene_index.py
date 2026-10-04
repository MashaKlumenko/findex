"""Build the Gutenberg index so Scalene can see tokenize and accumulate."""

from __future__ import annotations

import sys
from pathlib import Path

from findex.index import main

if __name__ == "__main__":
    out = Path(__import__("os").environ.get("TEMP", ".")) / "findex-scalene-index.pkl"
    sys.exit(main([str(Path("data/gutenberg")), "--out", str(out), "--representation", "slots"]))

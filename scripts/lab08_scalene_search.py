"""Repeat a resident search so Scalene can see the scorer, not just pickle.

The one-shot ``findex search`` command spends almost all of its time
unpickling ``Posting`` objects. This script loads once, then scores the
common term and the three-term query many times.
"""

from __future__ import annotations

from findex.rank import BM25, ranked_search
from findex.store import load


def main() -> None:
    index = load("data/index.pkl")
    scorer = BM25()
    for _ in range(200):
        ranked_search(index, "the", scorer=scorer, k=10, snippets=False)
    for _ in range(40):
        ranked_search(
            index,
            "elizabeth darcy marriage",
            scorer=scorer,
            k=10,
            snippets=False,
        )


if __name__ == "__main__":
    main()

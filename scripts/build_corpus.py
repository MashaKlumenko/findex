"""Build a local corpus under data/ from Project Gutenberg (public domain).

Downloads a handful of UTF-8 books, splits them into paragraph-sized
documents, and writes ``data/gutenberg/*.txt``. The result is gitignored.

    python scripts/build_corpus.py
"""

from __future__ import annotations

import ssl
import sys
import urllib.request
from pathlib import Path

# Plain-text UTF-8 mirrors. Public domain.
BOOKS = [
    ("pride-and-prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("alice-in-wonderland", "https://www.gutenberg.org/files/11/11-0.txt"),
    ("frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
    ("sherlock-holmes", "https://www.gutenberg.org/files/1661/1661-0.txt"),
    ("dorian-gray", "https://www.gutenberg.org/files/174/174-0.txt"),
    ("tale-of-two-cities", "https://www.gutenberg.org/files/98/98-0.txt"),
    ("huckleberry-finn", "https://www.gutenberg.org/files/76/76-0.txt"),
    ("dracula", "https://www.gutenberg.org/files/345/345-0.txt"),
]

MIN_CHARS = 280
USER_AGENT = "findex-lab01/0.1 (student corpus builder; +https://www.gutenberg.org)"


def fetch(url: str) -> str:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def paragraphs(text: str) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip():
            buf.append(line.strip())
            continue
        if buf:
            block = " ".join(buf)
            if len(block) >= MIN_CHARS:
                chunks.append(block)
            buf = []
    if buf:
        block = " ".join(buf)
        if len(block) >= MIN_CHARS:
            chunks.append(block)
    return chunks


def write_docs(root: Path, slug: str, chunks: list[str]) -> int:
    book_dir = root / slug
    book_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks, start=1):
        (book_dir / f"{i:04d}.txt").write_text(chunk + "\n", encoding="utf-8")
    return len(chunks)


def fallback_seed(root: Path, n: int = 1200) -> int:
    """If Gutenberg is unreachable, still produce a streamable corpus."""
    samples = [
        "The quick brown fox jumps over the lazy dog. Search engines count terms, not characters.",
        "Київ стоїть над Дніпром. П'ять яблук і добре відомий алгоритм токенізації.",
        "Café and cafe\u0301 must collapse to one token after NFC normalization.",
        "Generators yield one document at a time so a 2 GB corpus fits in a laptop's RAM.",
        "Don't materialize the stream: list(gen) is the bug this lab is about.",
    ]
    out = root / "seed"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        body = " ".join(samples[j % len(samples)] for j in range(i, i + 8))
        (out / f"{i:04d}.txt").write_text(body + f" document {i}\n", encoding="utf-8")
    return n


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    dest = project / "data" / "gutenberg"
    dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for slug, url in BOOKS:
        print(f"fetching {slug} ...", flush=True)
        try:
            text = fetch(url)
        except Exception as exc:  # noqa: BLE001 — network is optional
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        n = write_docs(dest, slug, paragraphs(text))
        print(f"  {n} documents")
        total += n

    if total < 200:
        print("Gutenberg download produced too little; writing seed corpus")
        total += fallback_seed(project / "data")

    print(f"done: {total} documents under {dest.parent}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())

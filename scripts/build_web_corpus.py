"""Sample the local Gutenberg corpus into demo/sample.jsonl for the image.

The full corpus stays gitignored. The sample is small enough to commit and
is what the Docker build indexes, so a deployed container does not need the
original files. Re-run this only when you want a fresher sample:

    python scripts/build_web_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

BOOKS = {
    "pride-and-prejudice": "Pride and Prejudice",
    "alice-in-wonderland": "Alice's Adventures in Wonderland",
    "frankenstein": "Frankenstein",
    "sherlock-holmes": "The Adventures of Sherlock Holmes",
    "dorian-gray": "The Picture of Dorian Gray",
    "tale-of-two-cities": "A Tale of Two Cities",
    "huckleberry-finn": "Adventures of Huckleberry Finn",
    "dracula": "Dracula",
}

PER_BOOK = 100
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "gutenberg"
TARGET = ROOT / "demo" / "sample.jsonl"


def heading(book: str, paragraph: str) -> str:
    sentence = paragraph.strip().split(". ", 1)[0]
    short = sentence if len(sentence) <= 48 else sentence[:48].rsplit(" ", 1)[0]
    line = f"{book} — {short}"
    return line[:80]


def sample_book(directory: Path, book: str) -> list[str]:
    files = sorted(directory.glob("*.txt"))
    if not files:
        return []
    step = max(1, len(files) // PER_BOOK)
    chosen = files[::step][:PER_BOOK]
    rows: list[str] = []
    for path in chosen:
        paragraph = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(paragraph) < 80:
            continue
        text = f"{heading(book, paragraph)}\n{paragraph}\n"
        rows.append(json.dumps({"text": text}, ensure_ascii=False))
    return rows


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"missing corpus at {SOURCE}; run scripts/build_corpus.py")
    rows: list[str] = []
    for slug, book in BOOKS.items():
        rows.extend(sample_book(SOURCE / slug, book))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} passages to {TARGET} ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

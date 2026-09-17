# findex

A search engine built one Python lab at a time. **Lab 1** is the intake: stream a corpus through generators so tokenization and term counts stay in constant memory.

> Loop like a native. The `for` loop is a protocol, not a counter.

## Corpus

Public-domain English novels from [Project Gutenberg](https://www.gutenberg.org/), split into paragraph-sized `.txt` files (one document per paragraph, ≥280 characters):

- Pride and Prejudice, Alice in Wonderland, Frankenstein, Sherlock Holmes,
  The Picture of Dorian Gray, A Tale of Two Cities, Huckleberry Finn, Dracula

That is **5 048 documents**, **~623k tokens**, under `data/` (gitignored). Rebuild with:

```bash
python scripts/build_corpus.py
```

If Gutenberg is unreachable, the script writes a smaller mixed English/Ukrainian seed corpus instead.

## Setup

The layout is `src/findex` as specified by the course (`uv init --lib`). `uv` is optional; plain Python 3.11+ is enough.

```bash
# with uv (course default)
uv init findex --lib   # already done in this repo
uv add --dev ruff pytest
uv run python -m findex.stats data/

# without uv
set PYTHONPATH=src          # Windows cmd
$env:PYTHONPATH = "src"     # PowerShell
export PYTHONPATH=src       # Unix
python -m findex.stats data/
python scripts/run_tokenize_checks.py
```

`--limit N` uses `itertools.islice` on the document stream so you can develop on the first N files:

```bash
python -m findex.stats data/ --limit 100
python -m findex.stats data/ --eager          # Lab 1 measurement only
```

## Tokenizer policy

`tokenize()` NFC-normalizes, then `casefold()`s, then streams `re.finditer` over

```text
\w+(?:['’-]\w+)*
```

| Topic | Choice |
|---|---|
| Apostrophes | Kept *inside* a token (`don't`, `п'ять`). Leading/trailing quotes are punctuation and drop. |
| Hyphens | Kept between word characters (`well-known`, `state-of-the-art`, `utf-8`). |
| Digits | Kept. Bare numbers and version-like tokens are useful in search. |
| Case | `casefold`, not `lower` — `"Straße"` → `strasse`. |
| Unicode | NFC so `café` and `cafe\u0301` become one token. `\w` is Unicode-aware, so Cyrillic matches. |

`re.finditer` is used instead of `findall` so a huge document is not turned into a list of every match before the caller asks for them.

## Pipeline

```text
iter_documents(root)  →  tokenize(doc.text)  →  Counter
     generator                 generator           sink
```

`iter_documents` walks a directory of `.txt` / `.md` files (`Path.rglob`) or a `.jsonl` file (one JSON object per line, `text` / `body` / `content`). Bad encodings use `errors="replace"` and unreadable files are logged and skipped.

## Eager vs lazy (same machine, warm disk cache)

Python 3.11.9, Windows 10, corpus = 5 048 Gutenberg paragraph files.

| Version | Documents | Peak memory | Elapsed |
|---|---|---|---|
| eager (lists) | 5048 | 48.5 MiB | 2.114 s |
| lazy (generators) | 5048 | 7.1 MiB | 2.605 s |

Peak memory is `tracemalloc.get_traced_memory()[1]`. Wall time is `time.perf_counter()`.

The eager path does `list(iter_documents(...))` and then `list(tokenize(doc.text))` for every document, so RAM holds every file body plus every token string at once, on top of the term `Counter`. The lazy path never keeps more than the current `Document` and the current token; the `Counter` (the vocabulary) is the structure that is *supposed* to grow, which is why peak memory is not zero. Time is similar once the 5 048 files are in the OS cache — generators are about memory, not a free speedup. The first cold run of the lazy pipeline took ~83 s, which was disk, not Python.

## Layout

```text
findex/
  pyproject.toml
  README.md
  .gitignore              # includes data/
  data/                   # corpus, not committed
  scripts/build_corpus.py
  src/findex/
    corpus.py             # iter_documents(root) -> Iterator[Document]
    tokenize.py           # tokenize(text) -> Iterator[str]
    stats.py              # python -m findex.stats
  tests/
    test_tokenize.py
    test_corpus.py
```

## What Lab 1 does *not* do yet

No inverted index, no ranking, no CLI package. Those are Labs 2–4. This repo is tagged `lab-01` when git is available.

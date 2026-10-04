# Lab 8 — profile before the rewrite

Measured on 4 Oct 2026, before any NumPy change. Python 3.13.16, GIL on, Windows, 13th Gen Intel Core i7-13650HX. Corpus: 5 048 Gutenberg passages, 24 006 terms, 623 261 tokens. The index on disk is the Lab 3 `slots` pickle (`data/index.pkl`, 409 457 `Posting` objects).

Flame graphs from `py-spy record -r 200` (sampling, not the cProfile instrumentation):

- [flame-search.svg](flame-search.svg) — `python -m findex search data/index.pkl "elizabeth darcy marriage" --limit 10 --no-snippets` (637 samples)
- [flame-index.svg](flame-index.svg) — `python -m findex index data/gutenberg --representation slots` (598 samples)

`cProfile` dumps are local only (`docs/cprofile-search.prof`, `docs/cprofile-index.prof`). Scalene 2.3.0, `--cpu-only`, line split of Python time vs native time is below. The three lines were written down before the scorer changed.

## Search

The one-shot CLI is not a scorer problem. Wall time without cProfile was **3.31 s to load** and **0.0005 s to score** the three-term query (8 hits). cProfile with the CLI's `tracemalloc` wrapped around the load stretched that load to 12.9 s; absolute cProfile seconds are inflated, the ranking is not.

| Rank | Where | What the tools say | Python or native | Hypothesis, before touching the formula |
|---|---|---|---|---|
| 1 | `store.py` `pickle.load` | Scalene on a resident search: **50%** of samples, of which **40% native** and **10% system**. cProfile tottime: `_pickle.load` 3.6 s, then `dataclasses.fields` 4.5 s and `_dataclass_setstate` 3.3 s, 409 457 calls each. The search flame graph is this bar. | Native pickle, plus a Python callback per object | 409k `Posting` instances. The bytes are not the cost; restoring a dataclass per posting is. An `int32` buffer should load like the Lab 2 `array('I')` row (0.057 s), not like a graph of objects. |
| 2 | `rank.py` `BM25.score` and the line that calls it | Scalene, 200 repeats of `"the"` after the load: **~11% + ~11%**, mostly **native** (`math.log`, float boxing). Own timing, snippets off, median of 9: `"the"` **4.94 ms** (4 812 docs), `"elizabeth darcy marriage"` **0.096 ms** (8 docs), `"anniversary"` **0.015 ms** (1 doc). | Native work *per posting*, driven by a Python loop | Scalene paints the line native because each iteration calls into C once. That is still the vectorization target: one ufunc over the posting array replaces thousands of those calls. The three-term query will barely move; its posting lists are short and the AND set is 8 documents. |
| 3 | `index.py` `iter_postings` yielding a `Posting` | Scalene **~6% native** on the `Posting(...)` line and **~3% Python** on `yield from`. | Python objects | The loop pays for an object even when the math has already left. `int32` doc ids and tfs remove the yield from the hot path. |

`make_snippet` was 0.086 s of the 0.110 s cProfile `ranked_search` when snippets were on (8 file reads and a regex). It is real, and it is not where a NumPy scorer helps. The table below times the scorer with snippets off.

## Index

| Rank | Where | What the tools say | Python or native | Hypothesis |
|---|---|---|---|---|
| 1 | `store.py` `pickle.dump` | Scalene **60%** (**51% native**, 9% system). cProfile cumtime 2.2 s, of which `dataclasses.fields` is 0.87 s tottime while building the pickle state. | Native dump, Python per object | Same 409k objects on the way out. NumPy postings shrink this. They do not make tokenization faster. |
| 2 | `index.py` document loop (`for offset, doc in enumerate(docs)`) | Scalene **37%**, split **Python 13% / native 13% / system 11%**. cProfile: `_accumulate_document` 1.31 s cum, `tokenize` 0.51 s cum (0.38 s tottime). | Mixed: disk (system), `re` (native), `Posting` construction (Python) | Reading and the regex are not a Python-loop bug. The Python slice is building one object per (term, doc) pair. An array append is the part worth changing; the regex stays. |
| 3 | `tokenize` / `re.Match.group` | cProfile tottime 0.38 s + 0.10 s. Scalene folds it into the loop line above. | Native regex, Python generator around it | Leave it. A vectorized scorer does not run during `findex index`. |

Build wall in this run, without the profiler: **1.57 s**, save **1.07 s**. Serial stays the default; Lab 5 already showed processes do not pay for themselves here.

## After the NumPy scorer

Same slots index, snippets off, one warmup (that warmup copies postings to `int32`), then the median of nine. Python 3.13.16, server stopped.

| Query | Matched | Lab 3 (ms) | NumPy (ms) | Speedup |
|---|---:|---:|---:|---:|
| `the` | 4812 | 3.549 | 0.221 | 16.0× |
| `anniversary` | 1 | 0.007 | 0.024 | 0.30× |
| `elizabeth darcy marriage` | 8 | 0.046 | 0.071 | 0.65× |

The common-term loop is the hotspot the profile pointed at, and it is the one that moved. A one-document list and an 8-document AND are shorter than the cost of building a mask and calling `argpartition`. Repeat with `python scripts/lab08_bench.py`.

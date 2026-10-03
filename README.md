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

## Лабораторна робота 2. Інвертований індекс: словники, хешування та пам'ять

### 1. Дослідження пам'яті (Memory Study)
Нижче наведено результати вимірювання пікової пам'яті (`tracemalloc`) під час побудови індексу, фінального розміру файлу на диску та часу завантаження для трьох різних представлень постінгів на одному і тому самому корпусі:

| Подання постінгів (Postings Representation) | Пікова пам'ять (Build) | Розмір файлу на диску | Час завантаження (Load) |
| :--- | :--- | :--- | :--- |
| `list[Posting]` з класичним `@dataclass` | 49.5 MiB | 9.5 MiB | 1.056 s |
| `list[Posting]` з `slots=True` | 34.1 MiB | 7.2 MiB | 1.327 s |
| `array('I')` contiguous buffers (без об'єктів) | **15.7 MiB** | **4.8 MiB** | **0.057 s** |

#### Куди пішли байти? (Аналіз споживання пам'яті)
* **Звичайний dataclass:** Споживає найбільше пам'яті (~49.5 MiB) через колосальний оверхед високорівневих об'єктів у Python. Кожен екземпляр класу за замовчуванням створює під капотом динамічний словник `__dict__` (хеш-таблицю) для зберігання своїх атрибутів. На мільйонах токенів/постінгів ці приховані хеш-таблиці та покажчики мови дублюють дані й марнують пам'ять.
* **Реалізація зі `slots=True`:** Прибирає динамічний словник `__dict__` з кожного об'єкта. Python виділяє під атрибути фіксований плоский масив фіксованого розміру відразу в пам'яті, що дало економію близько **30%** (пам'ять впала до 34.1 MiB, а файл зменшився на 2.3 MiB).
* **Масиви `array('I')`:** Найефективніший підхід. Ми повністю відмовляємося від створення об'єктів-обгорток Python для кожного окремого постінгу. Замість цього ідентифікатори документів зберігаються у вигляді неперервних (contiguous) бінарних буферів сирих `unsigned int` (чисел без знаку). Це знизило пікову пам'ять ще вдвічі (до **15.7 MiB**), а час завантаження скоротився майже в 20 разів (**0.057s**), оскільки операційній системі достатньо просто зчитати сирий шматок пам'яті в один крок.

---

### 2. Бенчмарк пошукових рушіїв (`merge` vs `set`)
Вимірювання проводились на найчастіших та найрідкісніших термінах корпусу:
* **Common terms:** 'and' (df=4815), 'the' (df=4812)
* **Rare terms:** 'anniversary' (df=1), 'enriching' (df=1)

| Тип операції | Запит | Рушій (`engine`) | Кількість хітів | Час виконання (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Common AND** | `and the` | merge | 4812 | 2.3571 ms |
| **Common AND** | `and the` | set | 4812 | 2.2537 ms |
| **Common OR** | `and OR the` | merge | 4812 | 2.1240 ms |
| **Common OR** | `and OR the` | set | 4812 | 2.0657 ms |
| **Rare AND** | `anniversary enriching` | merge | 0 | 1.9454 ms |
| **Rare AND** | `anniversary enriching` | set | 0 | 1.9530 ms |
| **Rare OR** | `anniversary OR enriching` | merge | 2 | 2.1096 ms |
| **Rare OR** | `anniversary OR enriching` | set | 2 | 2.0163 ms |

**Висновок:** Вбудований двигун `set` (множини Python) мінімально випереджає або йде нарівні з ручним алгоритмом `merge` (двох покажчиків). Це пов'язано з тим, що операції з множинами в Python (`&`, `|`) реалізовані на низькому рівні C за допомогою оптимізованих хеш-таблиць, тоді як покроковий обхід списків у `merge` виконується на рівні інтерпретатора Python і створює додатковий оверхед на ітерації циклів.

---

### 3. Порівняння серіалізації (`slots` варіант)
* **`pickle`**: розмір = 7.2 MiB | save = 1.084s | load = 1.230s
* **`json`**: розмір = 4.5 MiB | save = 0.648s | load = 0.952s

> ⚠️ **Безпека використання `pickle`:** Метод `pickle.load()` є фундаментально небезпечним для обробки неперевірених файлів з інтернету. Специфікація `pickle` дозволяє виконувати довільний Python-код під час десеріалізації об'єкта через магічний метод `__reduce__`. Зловмисник може підкинути шкідливий файл індексу, який при спробі пошуку виконає будь-які команди в терміналі жертви (Вразливість Remote Code Execution — RCE).

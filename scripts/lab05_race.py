"""Lost updates on a shared counter. The GIL does not protect this.

``counter += 1`` is LOAD_GLOBAL, LOAD_CONST, BINARY_OP, STORE_GLOBAL. A thread
switch between the load and the store drops an increment. A ``Lock`` makes the
read-modify-write one critical section. The indexer does not share a counter
at all: each worker returns its own partial index and the parent merges.

    python scripts/lab05_race.py
"""

from __future__ import annotations

import dis
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.resources import gil_enabled  # noqa: E402


def _bump_unlocked(n_threads: int, n_iters: int) -> int:
    counter = 0

    def work() -> None:
        nonlocal counter
        for _ in range(n_iters):
            counter += 1

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counter


def _bump_locked(n_threads: int, n_iters: int) -> int:
    counter = 0
    lock = threading.Lock()

    def work() -> None:
        nonlocal counter
        for _ in range(n_iters):
            with lock:
                counter += 1

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counter


def _show_bytecode() -> None:
    def inc() -> None:
        global counter
        counter += 1

    dis.dis(inc)


def main() -> int:
    n_threads = 8
    n_iters = 100_000
    expected = n_threads * n_iters
    print(f"python {sys.version.split()[0]}  sys._is_gil_enabled() = {gil_enabled()}")
    print("bytecode of `counter += 1`:")
    _show_bytecode()
    unlocked = _bump_unlocked(n_threads, n_iters)
    locked = _bump_locked(n_threads, n_iters)
    print(f"unlocked: {unlocked}   expected: {expected}   lost: {expected - unlocked}")
    print(f"locked:   {locked}   expected: {expected}   lost: {expected - locked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

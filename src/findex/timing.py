"""Closures-as-decorators used across the index and search path."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger("findex.timed")

F = TypeVar("F", bound=Callable)


def timed(fn: F) -> F:
    """Log wall time of ``fn`` in milliseconds. ``wraps`` keeps ``__name__``."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("%s took %.1f ms", fn.__name__, elapsed_ms)

    return wrapper  # type: ignore[return-value]

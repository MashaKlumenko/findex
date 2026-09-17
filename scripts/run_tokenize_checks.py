"""Run tokenizer asserts without pytest (Lab 4 will switch to pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.test_tokenize import (  # noqa: E402
    test_apostrophes_and_hyphens,
    test_casefold_eszett,
    test_combining_mark_accent,
    test_cyrillic,
    test_digits_are_kept,
    test_empty_string,
    test_mixed_case,
    test_punctuation_is_stripped,
    test_tokenize_is_a_generator,
)


def main() -> int:
    checks = [
        test_tokenize_is_a_generator,
        test_mixed_case,
        test_cyrillic,
        test_combining_mark_accent,
        test_punctuation_is_stripped,
        test_empty_string,
        test_apostrophes_and_hyphens,
        test_digits_are_kept,
        test_casefold_eszett,
    ]
    for fn in checks:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

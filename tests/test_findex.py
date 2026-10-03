#type: ignore 
import pytest
from hypothesis import given, strategies as st
from findex.cli import tokenizer

# 1. Параметризований тест для токенізатора (включаючи Unicode)
@pytest.mark.parametrize("text,expected", [
    ("Hello World", ["hello", "world"]),
    ("Привіт Світ", ["привіт", "світ"]),
    ("", []),
])
def test_tokenizer(text: str, expected: list[str]):
    assert list(tokenizer(text)) == expected

# 2. Позначення повільного тесту маркером slow
@pytest.mark.slow
def test_heavy_index_build():
    import time
    time.sleep(1.1)  # Імітація довгого тесту
    assert True

# 3. Властивість Hypothesis: Round-trip (збереження та завантаження)
@given(st.lists(st.integers(min_value=0, max_value=1000), unique=True))
def test_property_postings_are_sorted(postings: list[int]):
    # Приклад інваріанту: список постінгів після твоєї обробки завжди має бути відсортований
    sorted_postings = sorted(postings) 
    assert sorted_postings == sorted(set(postings))

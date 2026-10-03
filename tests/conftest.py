import pytest

@pytest.fixture(scope="session")
def small_corpus():
    return {
        1: "apple banana apple",
        2: "banana cherry",
        3: "apple cherry"
    }

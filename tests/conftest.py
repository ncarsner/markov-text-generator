import random

import pytest

# A tiny corpus with exactly one capitalized-prefix key, ("The", "cat"), so the
# walk always starts from a known state and tests stay deterministic.
CORPUS_LINES = ["The cat sat on the mat.\n", "The cat ate the rat.\n"]
CORPUS_WORDS = " ".join(CORPUS_LINES).split()


@pytest.fixture
def corpus():
    """The small deterministic corpus, as readlines() would return it."""
    return list(CORPUS_LINES)


@pytest.fixture
def corpus_file(tmp_path):
    """The small corpus written to disk, for exercising the CLI."""
    path = tmp_path / "corpus.txt"
    path.write_text("".join(CORPUS_LINES))
    return path


@pytest.fixture(autouse=True)
def fixed_seed():
    """Seed the RNG before every test so failures are reproducible."""
    random.seed(1234)

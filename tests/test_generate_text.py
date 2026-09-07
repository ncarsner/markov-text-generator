"""Tests for the pure chain-building/walking function."""

import random

import pytest

from markov_cli import generate_text

from conftest import CORPUS_WORDS


def source_pairs(words):
    """Every adjacent word pair that actually occurs in the source."""
    return set(zip(words, words[1:]))


def segments(raw_tokens):
    """Split a raw token stream on the empty strings that mark a chain reset."""
    out, current = [], []
    for token in raw_tokens:
        if token == "":
            if current:
                out.append(current)
            current = []
        else:
            current.append(token)
    if current:
        out.append(current)
    return out


class TestChainCorrectness:
    def test_every_word_comes_from_the_source(self, corpus):
        generated = generate_text(corpus, 50).split()
        assert set(generated) <= set(CORPUS_WORDS)

    def test_starts_from_a_capitalized_prefix(self, corpus):
        first = generate_text(corpus, 20).split()[0]
        assert first[:1].isupper()

    @pytest.mark.parametrize("seed", range(25))
    def test_adjacent_pairs_all_occur_in_the_source(self, corpus, seed):
        """The defining property of the chain: it never invents a transition.

        Checked per segment, because an empty token means the walk reset to the
        start of the corpus and the words either side of it were never adjacent.
        """
        random.seed(seed)
        pairs = source_pairs(CORPUS_WORDS)
        for segment in segments(generate_text(corpus, 60).split(" ")):
            for pair in zip(segment, segment[1:]):
                assert pair in pairs

    def test_treats_lines_as_one_continuous_stream(self):
        """A transition may span a line break: 'mat.' -> 'The' crosses lines."""
        lines = ["The cat sat on the mat.\n", "The end.\n"]
        random.seed(0)
        joined = " ".join(generate_text(lines, 200).split())
        assert "mat. The" in joined


class TestWordCount:
    def test_defaults_to_the_input_word_count(self, corpus):
        assert generate_text(corpus) is not None
        random.seed(1234)
        generated = generate_text(corpus)
        # Two seed words plus one per requested word, minus any empties that
        # split() collapses.
        assert len(generated.split()) <= len(CORPUS_WORDS) + 2

    def test_explicit_count_bounds_the_output(self, corpus):
        assert len(generate_text(corpus, 10).split()) <= 12

    def test_zero_is_not_a_sentinel_at_this_layer(self, corpus):
        """Passing 0 directly yields only the seed pair.

        The CLI translates 0 into None before calling in; the function itself
        does not. See test_cli.TestNumWords.test_zero_means_input_length.
        """
        assert generate_text(corpus, 0) == "The cat"


class TestDeterminism:
    def test_same_seed_gives_same_output(self, corpus):
        random.seed(99)
        first = generate_text(corpus, 30)
        random.seed(99)
        assert generate_text(corpus, 30) == first

    def test_different_seeds_can_diverge(self, corpus):
        outputs = set()
        for seed in range(30):
            random.seed(seed)
            outputs.add(generate_text(corpus, 30))
        assert len(outputs) > 1


class TestCorpusWrapAround:
    def test_end_of_corpus_loops_back_to_the_beginning(self):
        """Reaching the final word emits empties, then restarts the corpus.

        Documented, not endorsed: split() hides the seam, so printed output can
        silently splice the end of the source onto its start -- and on a short
        corpus the walk simply loops it over and over.
        """
        corpus = ["Alpha", "beta", "gamma."]
        raw = generate_text(["Alpha beta gamma.\n"], 12).split(" ")

        assert "" in raw, "expected empty tokens marking the reset"
        produced = segments(raw)
        assert len(produced) > 1, "expected the corpus to restart at least once"
        for segment in produced:
            assert segment == corpus[: len(segment)]

    def test_trailing_empties_show_up_as_trailing_whitespace(self):
        assert generate_text(["Hello"], 1) == "Hello  "


class TestEdgeCases:
    @pytest.mark.parametrize(
        "lines, label",
        [
            ([], "empty input"),
            ([""], "one blank line"),
            (["   \n"], "whitespace only"),
            (["the cat sat on the mat"], "no capitalized word"),
        ],
    )
    def test_no_capitalized_prefix_raises(self, lines, label):
        """Current behavior: an all-lowercase corpus crashes rather than degrading."""
        with pytest.raises(IndexError):
            generate_text(lines, 10)

    def test_single_word_corpus(self):
        """The only word available, repeated as the walk loops the corpus."""
        assert set(generate_text(["Hello"], 5).split()) == {"Hello"}

    def test_a_bare_string_is_iterated_by_character(self):
        """Footgun: the parameter is a list of lines, but a str is accepted silently."""
        generated = generate_text("The cat", 5)
        assert set(generated.split()) <= set("Thecat")

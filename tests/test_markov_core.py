"""Tests for the shared n-gram engine."""

import math
import random

import pytest

import markov_core
from markov_core import (
    NgramModel,
    build_chain,
    capitalized_contexts,
    chain_statistics,
    get_tokenizer,
    plain_tokens,
    sample,
    walk,
    whitespace_tokens,
)

CORPUS = "the cat sat on the mat the cat ate the rat".split()

# Cycles forever, so a walk never reaches a dead end. CORPUS does: it ends at
# "the rat", a context with no recorded continuation.
CYCLE = ["Alpha", "beta", "gamma"] * 4


class TestTokenizers:
    def test_whitespace_preserves_case_and_punctuation(self):
        assert whitespace_tokens("Freedom! is  not free") == ["Freedom!", "is", "not", "free"]

    def test_plain_lowercases_and_drops_punctuation(self):
        assert plain_tokens("Freedom! is not free.") == ["freedom", "is", "not", "free"]

    def test_plain_keeps_apostrophes_and_digits(self):
        """Contractions are single words; years are evidence, not noise."""
        assert plain_tokens("We don't, in 1962, agree") == ["we", "don't", "in", "1962", "agree"]

    def test_the_two_views_disagree_deliberately(self):
        text = "Freedom! Freedom."
        assert len(set(whitespace_tokens(text))) == 2   # generation keeps them distinct
        assert len(set(plain_tokens(text))) == 1        # analysis treats them as one

    def test_registry_lookup(self):
        assert get_tokenizer("plain") is plain_tokens
        assert get_tokenizer("whitespace") is whitespace_tokens

    def test_unknown_profile_names_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown tokenizer 'legal'"):
            get_tokenizer("legal")


class TestBuildChain:
    def test_maps_context_to_continuations(self):
        assert dict(build_chain(["a", "b", "a", "b", "c"], 2)) == {
            ("a", "b"): ["a", "c"],
            ("b", "a"): ["b"],
        }

    def test_keeps_duplicates_so_choice_is_frequency_weighted(self):
        chain = build_chain(["a", "b", "x", "a", "b", "x", "a", "b", "y"], 2)
        assert chain[("a", "b")] == ["x", "x", "y"]

    def test_order_is_a_parameter(self):
        assert list(build_chain(CORPUS, 3))[0] == ("the", "cat", "sat")

    def test_corpus_shorter_than_order_yields_nothing(self):
        assert dict(build_chain(["a", "b"], 3)) == {}

    def test_rejects_order_below_one(self):
        with pytest.raises(ValueError, match="at least 1"):
            build_chain(CORPUS, 0)

    def test_missing_context_yields_an_empty_list(self):
        """defaultdict semantics: a dead end is an empty list, not a KeyError."""
        assert build_chain(CORPUS, 2)[("no", "such")] == []


class TestCapitalizedContexts:
    def test_selects_contexts_opening_with_a_capital(self):
        chain = build_chain(["A", "b", "A", "b", "c"], 2)
        assert capitalized_contexts(chain) == [("A", "b")]

    def test_empty_when_nothing_is_capitalized(self):
        assert capitalized_contexts(build_chain(CORPUS, 2)) == []


class TestSample:
    def test_returns_start_plus_count_tokens(self):
        out = sample(build_chain(CYCLE, 2), ("Alpha", "beta"), 5, 2)
        assert len(out) == 7
        assert out[:2] == ["Alpha", "beta"]

    def test_only_follows_recorded_transitions(self):
        chain = build_chain(CYCLE, 2)
        out = sample(chain, ("Alpha", "beta"), 6, 2)
        for i in range(len(out) - 2):
            assert out[i + 2] in chain[(out[i], out[i + 1])]

    def test_is_reproducible_with_a_seeded_generator(self):
        chain = build_chain(CYCLE, 2)
        first = sample(chain, ("Alpha", "beta"), 10, 2, rng=random.Random(7))
        assert sample(chain, ("Alpha", "beta"), 10, 2, rng=random.Random(7)) == first

    def test_running_off_the_end_of_the_corpus_raises(self):
        """CORPUS ends at 'the rat'; nothing follows it."""
        with pytest.raises(IndexError):
            sample(build_chain(CORPUS, 2), ("the", "rat"), 1, 2)

    def test_dead_end_raises(self):
        with pytest.raises(IndexError):
            sample(build_chain(CORPUS, 2), ("no", "such"), 1, 2)


class TestWalk:
    def test_yields_only_the_tokens_after_the_start(self):
        out = list(walk(build_chain(CYCLE, 2), ("Alpha", "beta"), 2, limit=4))
        assert out == ["gamma", "Alpha", "beta", "gamma"]

    def test_stops_at_the_end_of_the_corpus_instead_of_raising(self):
        """The difference from sample(). CORPUS ends at 'the rat'."""
        assert list(walk(build_chain(CORPUS, 2), ("the", "rat"), 2, limit=10)) == []

    def test_a_short_walk_is_not_padded(self):
        """'ate the rat' ends the corpus, so the walk yields what it can."""
        out = list(walk(build_chain(CORPUS, 2), ("cat", "ate"), 2, limit=10))
        assert out == ["the", "rat"]

    def test_no_limit_runs_until_the_chain_stops(self):
        assert len(list(walk(build_chain(CORPUS, 2), ("cat", "ate"), 2))) == 2

    def test_stop_ends_the_walk_and_keeps_the_token_that_ended_it(self):
        out = list(
            walk(
                build_chain(CYCLE, 2),
                ("Alpha", "beta"),
                2,
                limit=100,
                stop=lambda word: word == "Alpha",
            )
        )
        assert out == ["gamma", "Alpha"]

    def test_stop_is_checked_after_the_limit_is_reached(self):
        """limit bounds the walk even when stop never fires."""
        out = list(
            walk(
                build_chain(CYCLE, 2),
                ("Alpha", "beta"),
                2,
                limit=3,
                stop=lambda word: word == "never",
            )
        )
        assert len(out) == 3

    def test_does_not_grow_the_chain_it_walks(self):
        """A defaultdict inserts on lookup; walking a dead end must not."""
        chain = build_chain(CORPUS, 2)
        before = len(chain)
        list(walk(chain, ("no", "such"), 2, limit=5))
        assert len(chain) == before

    def test_a_stop_condition_without_a_limit_is_refused(self):
        """On CYCLE, a condition that never fires is an unbounded walk. The
        guard refuses at the call, before any walking, which is why this is
        checked on the finite CORPUS: without the guard the walk here ends on
        its own, so a missing guard fails this test instead of hanging it."""
        with pytest.raises(ValueError, match="limit"):
            list(walk(build_chain(CORPUS, 2), ("the", "cat"), 2,
                      stop=lambda word: word == "never"))

    def test_is_reproducible_with_a_seeded_generator(self):
        chain = build_chain(CYCLE, 2)
        args = (chain, ("Alpha", "beta"), 2)
        first = list(walk(*args, rng=random.Random(3), limit=12))
        assert list(walk(*args, rng=random.Random(3), limit=12)) == first


class TestNgramModelCounts:
    def test_records_totals_and_vocabulary(self):
        model = NgramModel(CORPUS, order=3)
        assert model.total == 11
        assert model.vocab == 7

    def test_counts_are_indexed_by_n(self):
        model = NgramModel(CORPUS, order=3)
        assert model.counts[1][("the",)] == 4
        assert model.counts[2][("the", "cat")] == 2
        assert model.counts[3][("the", "cat", "sat")] == 1

    def test_rejects_order_below_one(self):
        with pytest.raises(ValueError, match="at least 1"):
            NgramModel(CORPUS, order=0)

    def test_accepts_any_iterable(self):
        assert NgramModel(iter(CORPUS), order=2).total == 11


class TestBackoff:
    """The backoff level is what makes a flagged span explainable."""

    @pytest.fixture
    def model(self):
        return NgramModel(CORPUS, order=3)

    @pytest.mark.parametrize(
        "context, word, level, count, label",
        [
            (("the", "cat"), "sat", 2, 1, "trigram hit"),
            (("xyz", "cat"), "sat", 1, 1, "backs off to bigram"),
            ((), "the", 0, 4, "unigram only"),
            (("the", "cat"), "zebra", -1, 0, "out of vocabulary"),
        ],
    )
    def test_reports_the_level_it_matched_at(self, model, context, word, level, count, label):
        _, actual_level, actual_count = model.score(context, word)
        assert (actual_level, actual_count) == (level, count)

    def test_backing_off_costs_a_penalty(self, model):
        """The same count reached by backoff must score lower than a direct hit."""
        direct, _, _ = model.score(("the", "cat"), "sat")
        backed_off, _, _ = model.score(("xyz", "cat"), "sat")
        assert backed_off < direct

    def test_unseen_words_score_above_zero(self, model):
        """No zero probabilities, so surprisal is always finite."""
        assert model.probability(("the", "cat"), "zebra") > 0
        assert math.isfinite(model.surprisal(("the", "cat"), "zebra"))

    def test_context_longer_than_the_order_is_truncated(self, model):
        long_context = ("a", "b", "c", "d", "the", "cat")
        assert model.score(long_context, "sat") == model.score(("the", "cat"), "sat")


class TestSurprisal:
    @pytest.fixture
    def model(self):
        return NgramModel(CORPUS, order=3)

    def test_a_certain_continuation_costs_nothing(self, model):
        """'on' only ever follows 'cat sat', so it carries no surprise."""
        assert model.surprisal(("cat", "sat"), "on") == pytest.approx(0, abs=1e-12)

    def test_unexpected_words_cost_more_than_expected_ones(self, model):
        assert model.surprisal(("the", "cat"), "zebra") > model.surprisal(("the", "cat"), "sat")

    def test_one_bit_means_twice_as_unexpected(self, model):
        """The scale is logarithmic; this is what the README explains."""
        likely = model.probability(("the", "cat"), "sat")
        assert model.surprisal(("the", "cat"), "sat") == pytest.approx(-math.log2(likely))

    def test_surprisals_align_with_the_document(self, model):
        assert len(model.surprisals(CORPUS)) == len(CORPUS)

    def test_bits_per_token_is_the_mean_not_the_peak(self, model):
        """A single alarming token must not set the document score."""
        scores = model.surprisals(CORPUS)
        assert model.bits_per_token(CORPUS) == pytest.approx(sum(scores) / len(scores))
        assert model.bits_per_token(CORPUS) < max(scores)

    def test_a_corpus_is_less_surprising_to_its_own_model(self, model):
        foreign = "zebra quagga okapi wander the veldt".split()
        assert model.bits_per_token(CORPUS) < model.bits_per_token(foreign)

    def test_empty_document_raises(self, model):
        with pytest.raises(ValueError, match="empty document"):
            model.bits_per_token([])


class TestPerplexity:
    def test_is_two_to_the_bits_per_token(self):
        model = NgramModel(CORPUS, order=3)
        assert model.perplexity(CORPUS) == pytest.approx(2 ** model.bits_per_token(CORPUS))


class TestNovelNgramRate:
    @pytest.fixture
    def model(self):
        return NgramModel(CORPUS, order=3)

    def test_zero_for_the_reference_itself(self, model):
        assert model.novel_ngram_rate(CORPUS) == 0.0

    def test_one_for_entirely_novel_text(self, model):
        assert model.novel_ngram_rate("zebra quagga okapi giraffe".split()) == 1.0

    def test_n_is_selectable(self, model):
        assert model.novel_ngram_rate(CORPUS, n=1) == 0.0

    @pytest.mark.parametrize("n", [0, 4])
    def test_rejects_n_outside_the_model(self, model, n):
        with pytest.raises(ValueError, match="between 1 and 3"):
            model.novel_ngram_rate(CORPUS, n=n)

    def test_document_shorter_than_n_raises(self, model):
        with pytest.raises(ValueError, match="shorter than"):
            model.novel_ngram_rate(["one", "two"], n=3)


class TestChainStatistics:
    def test_counts_distinct_continuations_not_repeats(self):
        branching, forced = chain_statistics(["a", "b", "c", "a", "b", "d"], 2)
        assert branching == pytest.approx(4 / 3)   # ('a','b') has 2, others 1 each
        assert forced == pytest.approx(2 / 3)

    def test_a_corpus_with_no_repetition_is_entirely_forced(self):
        _, forced = chain_statistics("a b c d e f".split(), 2)
        assert forced == 1.0

    def test_corpus_shorter_than_the_order_raises(self):
        with pytest.raises(ValueError, match="shorter than"):
            chain_statistics(["a", "b"], 3)


class TestConstants:
    def test_backoff_alpha_matches_the_documented_value(self):
        """0.4 is the value the PRD and README commit to; changing it changes scores."""
        assert markov_core.BACKOFF_ALPHA == 0.4

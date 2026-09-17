"""Tests for the generate subcommand: length, stopping, seeding, and export.

Generation is the by-product of the counts that analysis is built on, but it is
a separate code path from markov.py, which keeps its original behavior. The
distinguishing cases are here: the corpus ends rather than looping, an
all-lowercase corpus works rather than crashing, and a seed reproduces a run.
"""

import random
import subprocess
import sys

import pytest

import markov_cli
from markov_cli import (
    ends_sentence,
    generate_tokens,
    main,
    opening_context,
)

import markov_core

# Four sentences, all opening with a capital, so a sentence-start opening and a
# capitalized-word opening are distinguishable from a mid-sentence one.
PROSE = (
    "The cat sat on the mat. The cat ate the rat. "
    "The dog sat on the log. The dog ate the frog.\n"
)


# Two sentences that lead back into each other, so a walk from ("The", "cat")
# never runs out. Needed wherever "stopped at exactly N" has to be distinguished
# from "the corpus ended at N".
ENDLESS = "The cat sat. The cat ran. " * 6


@pytest.fixture
def prose(tmp_path):
    path = tmp_path / "prose.txt"
    path.write_text(PROSE, encoding="utf-8")
    return path


def chain_of(text, order=2):
    tokens = markov_core.whitespace_tokens(text)
    return tokens, markov_core.build_chain(tokens, order)


class TestSentenceDetection:
    @pytest.mark.parametrize("token", ["mat.", "now!", "why?", 'end."', "yes?)", "it.”"])
    def test_closing_punctuation_ends_a_sentence(self, token):
        assert ends_sentence(token)

    @pytest.mark.parametrize("token", ["cat", "mid-sentence,", "a.b", "3.14"])
    def test_other_tokens_do_not(self, token):
        assert not ends_sentence(token)

    def test_an_abbreviation_is_read_as_a_sentence_end(self):
        """A known limitation, not an oversight: 'Mr.' is indistinguishable
        from a full stop without a dictionary, and --sentences will overcount
        on prose full of abbreviations."""
        assert ends_sentence("Mr.")


class TestOpeningContext:
    def test_prefers_a_context_that_follows_a_sentence_end(self):
        tokens, chain = chain_of(PROSE)
        openings = {
            opening_context(tokens, chain, 2, random.Random(seed))
            for seed in range(40)
        }
        assert openings == {("The", "cat"), ("The", "dog")}

    def test_a_capitalized_word_mid_sentence_is_not_an_opening(self):
        """The distinction from markov.py's seeding: it would start at
        ('American', 'sound.') and emit a fragment."""
        text = "we heard the American sound. we heard the American noise."
        tokens, chain = chain_of(text)
        openings = {
            opening_context(tokens, chain, 2, random.Random(seed))
            for seed in range(20)
        }
        assert openings == {("we", "heard")}

    def test_the_corpus_opening_is_an_opening(self):
        """Nothing precedes the first context, so no sentence end points at it;
        it is still where the document itself begins.

        The corpus deliberately does not end in punctuation. Where it does,
        tokens[i - 1] at i == 0 wraps to that final token and admits the first
        context by accident, which would hide a missing guard here.
        """
        text = "Alpha beta gamma. The cat sat"
        tokens, chain = chain_of(text)
        openings = {
            opening_context(tokens, chain, 2, random.Random(seed))
            for seed in range(30)
        }
        assert ("Alpha", "beta") in openings
        assert openings == {("Alpha", "beta"), ("The", "cat")}

    def test_falls_back_when_nothing_ends_a_sentence(self):
        tokens, chain = chain_of("the cat sat on the mat")
        assert opening_context(tokens, chain, 2, random.Random(0)) in chain

    def test_an_empty_chain_is_reported_not_crashed(self):
        with pytest.raises(SystemExit) as excinfo:
            opening_context([], {}, 2, random.Random(0))
        assert "no word sequences" in str(excinfo.value)


class TestLength:
    def test_words_counts_the_whole_output(self):
        """--words N means N words, opening context included."""
        tokens, chain = chain_of(PROSE)
        produced, _ = generate_tokens(
            chain, ("The", "cat"), 2, random.Random(0), words=9
        )
        assert len(produced) == 9

    @pytest.mark.parametrize("wanted", [1, 2, 3, 7])
    def test_a_word_count_below_the_opening_trims_it(self, wanted):
        """The opening context is two tokens; --words 1 must still mean one."""
        tokens, chain = chain_of(ENDLESS)
        produced, note = generate_tokens(
            chain, ("The", "cat"), 2, random.Random(0), words=wanted
        )
        assert len(produced) == wanted
        assert produced[:min(wanted, 2)] == ["The", "cat"][:wanted]
        assert note is None

    @pytest.mark.parametrize("wanted", [1, 2, 3, 5])
    def test_sentences_stops_on_the_requested_count(self, wanted):
        """Checked on a corpus that never runs out, so stopping at exactly N is
        the reason the walk ended -- not the corpus ending at N anyway."""
        tokens, chain = chain_of(ENDLESS)
        produced, note = generate_tokens(
            chain, ("The", "cat"), 2, random.Random(0), sentences=wanted
        )
        assert sum(ends_sentence(word) for word in produced) == wanted
        assert produced[-1].endswith(".")
        assert note is None

    def test_the_sentence_cap_bounds_a_corpus_with_no_punctuation(self, monkeypatch):
        monkeypatch.setattr(markov_cli, "SENTENCE_TOKEN_LIMIT", 30)
        tokens, chain = chain_of("a b a b a b a b")
        produced, note = generate_tokens(
            chain, ("a", "b"), 2, random.Random(0), sentences=3
        )
        assert len(produced) == 32
        assert "reached the 30-token limit" in note


class TestRunningOutOfCorpus:
    def test_the_walk_ends_rather_than_looping(self):
        """markov.py splices the end of the corpus onto its beginning; this
        stops. Nothing after 'the frog.' was ever recorded."""
        tokens, chain = chain_of(PROSE)
        produced, note = generate_tokens(
            chain, ("The", "cat"), 2, random.Random(0), words=500
        )
        assert produced[-1] == "frog."
        assert "corpus ran out" in note

    def test_a_short_run_is_reported_not_hidden(self):
        tokens, chain = chain_of(PROSE)
        _, note = generate_tokens(
            chain, ("The", "cat"), 2, random.Random(0), sentences=99
        )
        assert "of 99 sentences" in note
        assert "corpus ran out" in note

    def test_a_complete_run_has_nothing_to_report(self):
        tokens, chain = chain_of(PROSE)
        _, note = generate_tokens(
            chain, ("The", "cat"), 2, random.Random(0), words=6
        )
        assert note is None


class TestChainCorrectness:
    def test_every_transition_occurs_in_the_source(self, prose, capsys):
        main(["generate", "--reference", str(prose), "--words", "60", "--seed", "5"])
        produced = capsys.readouterr().out.split()
        source = markov_core.whitespace_tokens(PROSE)
        pairs = set(zip(source, source[1:]))
        for pair in zip(produced, produced[1:]):
            assert pair in pairs

    def test_no_empty_tokens_leak_into_the_output(self, prose, capsys):
        """The sentinel that makes markov.py loop is absent here."""
        main(["generate", "--reference", str(prose), "--words", "40", "--seed", "2"])
        assert "" not in capsys.readouterr().out.split(" ")

    def test_order_is_honored(self, prose, capsys):
        main([
            "generate", "--reference", str(prose),
            "--order", "3", "--words", "30", "--seed", "1",
        ])
        produced = capsys.readouterr().out.split()
        source = markov_core.whitespace_tokens(PROSE)
        triples = set(zip(source, source[1:], source[2:]))
        for triple in zip(produced, produced[1:], produced[2:]):
            assert triple in triples


class TestDeterminism:
    def test_the_same_seed_reproduces_a_run(self, prose, capsys):
        argv = ["generate", "--reference", str(prose), "--words", "30", "--seed", "8"]
        main(argv)
        first = capsys.readouterr().out
        main(argv)
        assert capsys.readouterr().out == first

    def test_seeding_does_not_depend_on_the_global_rng(self, prose, capsys):
        """--seed must be self-contained: a caller that seeded random itself
        differently must still get the same text."""
        argv = ["generate", "--reference", str(prose), "--words", "30", "--seed", "8"]
        random.seed(1)
        main(argv)
        first = capsys.readouterr().out
        random.seed(999)
        main(argv)
        assert capsys.readouterr().out == first

    def test_different_seeds_can_diverge(self, prose, capsys):
        outputs = set()
        for seed in range(12):
            main([
                "generate", "--reference", str(prose),
                "--words", "20", "--seed", str(seed),
            ])
            outputs.add(capsys.readouterr().out)
        assert len(outputs) > 1


class TestOutputFile:
    def test_output_writes_the_text_to_the_file(self, prose, tmp_path, capsys):
        destination = tmp_path / "generated.txt"
        main([
            "generate", "--reference", str(prose),
            "--words", "5", "--seed", "4", "--output", str(destination),
        ])
        written = destination.read_text(encoding="utf-8").split()
        assert len(written) == 5
        assert set(written) <= set(markov_core.whitespace_tokens(PROSE))

    def test_output_leaves_stdout_empty(self, prose, tmp_path, capsys):
        """A pipeline reading stdout must not be handed a status line."""
        destination = tmp_path / "generated.txt"
        main([
            "generate", "--reference", str(prose),
            "--words", "20", "--seed", "4", "--output", str(destination),
        ])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert str(destination) in captured.err

    def test_the_file_matches_what_stdout_would_have_shown(self, prose, tmp_path, capsys):
        argv = ["generate", "--reference", str(prose), "--words", "20", "--seed", "4"]
        main(argv)
        printed = capsys.readouterr().out
        destination = tmp_path / "generated.txt"
        main(argv + ["--output", str(destination)])
        capsys.readouterr()
        assert destination.read_text(encoding="utf-8") == printed

    def test_the_file_is_written_as_utf8(self, tmp_path, capsys):
        source = tmp_path / "smart.txt"
        source.write_text("The “new” order — begins. The old order — ends.\n",
                          encoding="utf-8")
        destination = tmp_path / "generated.txt"
        main([
            "generate", "--reference", str(source),
            "--words", "8", "--seed", "0", "--output", str(destination),
        ])
        assert "—" in destination.read_text(encoding="utf-8")

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="EncodingWarning added in 3.10"
    )
    def test_the_write_encoding_is_never_left_to_the_locale(self, tmp_path):
        """Writing an em-dash proves nothing on a UTF-8 machine. Re-run under
        -X warn_default_encoding, where an implicit open() is an error, so this
        fails on the machine where the bug would actually bite -- the same
        guard test_cli applies to the read side."""
        source = tmp_path / "smart.txt"
        source.write_text("The “new” order — begins. The old order — ends.\n",
                          encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-X", "warn_default_encoding",
                "-W", "error::EncodingWarning",
                "markov_cli.py", "generate",
                "--reference", str(source),
                "--words", "8", "--seed", "0",
                "--output", str(tmp_path / "generated.txt"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestNotes:
    def test_a_short_run_is_reported_on_stderr(self, prose, capsys):
        main(["generate", "--reference", str(prose), "--words", "500", "--seed", "1"])
        captured = capsys.readouterr()
        assert "note: the corpus ran out" in captured.err
        assert "note:" not in captured.out


class TestLowercaseCorpus:
    def test_an_all_lowercase_corpus_generates_rather_than_crashing(
        self, tmp_path, capsys
    ):
        """markov.py raises IndexError here; see
        test_generate_text.TestEdgeCases.test_no_capitalized_prefix_raises."""
        source = tmp_path / "lower.txt"
        source.write_text("the cat sat on the mat and the cat ate the rat\n")
        assert main([
            "generate", "--reference", str(source), "--words", "8", "--seed", "1",
        ]) == 0
        assert capsys.readouterr().out.strip()


class TestValidation:
    def test_words_and_sentences_are_mutually_exclusive(self, prose):
        with pytest.raises(SystemExit) as excinfo:
            main([
                "generate", "--reference", str(prose),
                "--words", "10", "--sentences", "2",
            ])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("flag", ["--words", "--sentences", "--order"])
    def test_counts_below_one_exit_2(self, prose, flag):
        with pytest.raises(SystemExit) as excinfo:
            main(["generate", "--reference", str(prose), flag, "0"])
        assert excinfo.value.code == 2

    def test_a_corpus_shorter_than_the_order_is_named(self, tmp_path):
        source = tmp_path / "tiny.txt"
        source.write_text("a b c\n")
        with pytest.raises(SystemExit) as excinfo:
            main(["generate", "--reference", str(source), "--order", "5"])
        assert "too short" in str(excinfo.value)

    def test_a_missing_corpus_is_named(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["generate", "--reference", str(tmp_path / "nope.txt")])
        assert "nope.txt" in str(excinfo.value)

    def test_generate_requires_a_reference(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["generate"])
        assert excinfo.value.code == 2


class TestDefaults:
    def test_defaults_to_100_words(self, capsys):
        """Checked against a corpus long enough not to run out first."""
        import glob

        corpus = sorted(glob.glob("data/input/speech_inaugural_*.txt"))
        if not corpus:
            pytest.skip("corpus not present; run scripts/fetch_corpus.py")
        main(["generate", "--reference", corpus[0], "--seed", "0"])
        assert len(capsys.readouterr().out.split()) == markov_cli.GENERATE_WORDS

    def test_defaults_to_order_2(self, prose, capsys):
        argv = ["generate", "--reference", str(prose), "--words", "25", "--seed", "6"]
        main(argv)
        default = capsys.readouterr().out
        main(argv + ["--order", str(markov_cli.GENERATE_ORDER)])
        assert capsys.readouterr().out == default

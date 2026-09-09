"""Tests for span localization -- the report's actual deliverable."""

import glob
import json
import os

import pytest

import markov_cli
import markov_core
from markov_cli import (
    document_spans,
    main,
    rank_spans,
    span_surprisals,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INAUGURALS = sorted(glob.glob(os.path.join(REPO, "data/input/speech_inaugural_*.txt")))
MOON = os.path.join(REPO, "data/input/speech_we_choose_to_go_to_the_moon.txt")

requires_corpus = pytest.mark.skipif(
    len(INAUGURALS) < 54 or not os.path.exists(MOON),
    reason="public-domain corpus absent; run scripts/fetch_corpus.py",
)

REFERENCE = [
    ("a.txt", "the cat sat on the mat and the dog sat on the log\n"),
    ("b.txt", "the dog ate the rat and the cat ate the mat\n"),
    ("c.txt", "the cat sat on the log and the rat sat on the mat\n"),
]


class TestLocateTokens:
    @pytest.mark.parametrize("name", ["plain", "whitespace"])
    def test_produces_exactly_the_tokenizer_s_tokens(self, name):
        text = "The Cat sat!  On the mat, in 1962.\nThe dog didn't.\n"
        located = [token for token, _, _ in markov_core.locate_tokens(text, name)]
        assert located == markov_core.TOKENIZERS[name](text)

    def test_offsets_point_back_at_the_source(self):
        text = "The Cat sat on the mat.\n"
        for token, start, end in markov_core.locate_tokens(text, "whitespace"):
            assert text[start:end] == token

    def test_plain_lowercases_without_disturbing_offsets(self):
        """Lowercasing the text first would be simpler and wrong: it is not
        guaranteed to preserve length, and the offsets are what let a span
        quote the document as written."""
        text = "The CAT sat.\n"
        located = markov_core.locate_tokens(text, "plain")
        assert [token for token, _, _ in located] == ["the", "cat", "sat"]
        assert [text[start:end] for _, start, end in located] == ["The", "CAT", "sat"]

    def test_unknown_tokenizer_is_refused(self):
        with pytest.raises(ValueError, match="unknown tokenizer"):
            markov_core.locate_tokens("text", "legal")


class TestLineNumbers:
    def test_offsets_map_to_one_based_lines(self):
        text = "first line\nsecond line\nthird line\n"
        starts = markov_core.line_starts(text)
        assert markov_core.line_of(starts, 0) == 1
        assert markov_core.line_of(starts, text.index("second")) == 2
        assert markov_core.line_of(starts, text.index("third")) == 3

    def test_the_last_character_of_a_line_stays_on_that_line(self):
        text = "first line\nsecond line\n"
        starts = markov_core.line_starts(text)
        assert markov_core.line_of(starts, text.index("\n") - 1) == 1


class TestSpanSurprisals:
    def test_each_span_is_the_mean_of_its_window(self):
        assert span_surprisals([1.0, 2.0, 3.0, 4.0], 2) == [1.5, 2.5, 3.5]

    def test_one_span_per_starting_position(self):
        assert len(span_surprisals([1.0] * 10, 4) ) == 7

    def test_a_document_shorter_than_the_window_has_no_spans(self):
        assert span_surprisals([1.0, 2.0], 5) == []


class TestRanking:
    def test_the_highest_scoring_span_comes_first(self):
        means = [1.0, 9.0, 2.0, 3.0]
        assert rank_spans(means, 1, 2) == [1, 3]

    def test_overlapping_spans_are_suppressed(self):
        """Neighbouring windows share all but one token and score alike; a raw
        ranking would return the same passage a dozen times, one token over."""
        means = [9.0, 8.9, 8.8, 1.0, 5.0]
        assert rank_spans(means, 3, 2) == [0, 4]

    def test_ties_are_broken_by_position(self):
        assert rank_spans([5.0, 5.0, 5.0], 1, 3) == [0, 1, 2]

    def test_asking_for_more_spans_than_exist_returns_what_there_is(self):
        assert rank_spans([9.0, 8.0], 2, 5) == [0]

    def test_ranking_the_negated_scores_finds_the_least_variant(self):
        means = [9.0, 1.0, 8.0]
        assert rank_spans([-m for m in means], 1, 1) == [1]


class TestDocumentSpans:
    @pytest.fixture
    def scored(self):
        text = (
            "the cat sat on the mat and the dog sat on the log\n"
            "a bird flew over the shining sea beyond the distant hill\n"
        )
        model = markov_core.NgramModel(
            [tok for _, t in REFERENCE for tok in markov_core.plain_tokens(t)],
            order=2,
        )
        located = markov_core.locate_tokens(text, "plain")
        tokens = [token for token, _, _ in located]
        return text, located, model.surprisals(tokens)

    def test_reports_both_ends_of_the_ranking(self, scored):
        text, located, surprisals = scored
        spans = document_spans(text, located, surprisals, 6, 2)
        assert len(spans["most_variant"]) == 2
        assert len(spans["least_variant"]) == 2

    def test_the_unlike_passage_outranks_the_familiar_one(self, scored):
        text, located, surprisals = scored
        spans = document_spans(text, located, surprisals, 6, 1)

        # Line 1 repeats the reference's wording; line 2 is unlike it.
        assert spans["most_variant"][0]["first_line"] == 2
        assert "bird" in spans["most_variant"][0]["text"]
        assert spans["least_variant"][0]["first_line"] == 1
        assert "bird" not in spans["least_variant"][0]["text"]

    def test_a_span_quotes_the_source_as_written(self, scored):
        """The model's view has dropped capitals and punctuation; a reviewer
        has to recognize the passage in the document."""
        text = "The CAT sat on the mat, and the dog sat on the log.\n"
        located = markov_core.locate_tokens(text, "plain")
        model = markov_core.NgramModel(
            [tok for _, t in REFERENCE for tok in markov_core.plain_tokens(t)],
            order=2,
        )
        spans = document_spans(
            text, located, model.surprisals([t for t, _, _ in located]), 6, 1
        )
        quoted = spans["most_variant"][0]["text"]
        assert quoted in text
        assert quoted[0].isupper() or "," in quoted

    def test_spans_report_the_lines_they_came_from(self, scored):
        text, located, surprisals = scored
        spans = document_spans(text, located, surprisals, 6, 2)
        lines = [s["first_line"] for s in spans["most_variant"]]
        assert all(line >= 1 for line in lines)
        assert 2 in [s["first_line"] for s in spans["most_variant"]]

    def test_a_document_shorter_than_the_window_has_no_spans(self, scored):
        text, located, surprisals = scored
        assert document_spans(text, located, surprisals, 500, 5)["most_variant"] == []

    def test_zero_spans_requested_reports_none(self, scored):
        text, located, surprisals = scored
        assert document_spans(text, located, surprisals, 6, 0)["most_variant"] == []


class TestSpanReport:
    @pytest.fixture
    def reference_dir(self, tmp_path):
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)
        return reference

    @pytest.fixture
    def target(self, tmp_path):
        path = tmp_path / "target.txt"
        path.write_text(
            "the cat sat on the mat and the dog sat on the log\n"
            "a bird flew over the shining sea beyond the distant hill\n"
        )
        return path

    def test_the_report_names_the_window_and_lists_both_ends(
        self, reference_dir, target, capsys
    ):
        main(["analyze", str(target), "--reference", str(reference_dir),
              "--order", "2", "--window", "6", "--top-spans", "2"])
        out = capsys.readouterr().out
        assert "most variant spans (6-token window)" in out
        assert "least variant spans:" in out

    def test_a_span_on_one_line_is_cited_as_one_line(
        self, reference_dir, target, capsys
    ):
        main(["analyze", str(target), "--reference", str(reference_dir),
              "--order", "2", "--window", "6", "--top-spans", "1"])
        out = capsys.readouterr().out
        assert "  l." in out and "ll." not in out

    def test_a_span_crossing_lines_is_cited_as_a_range(
        self, reference_dir, tmp_path, capsys
    ):
        target = tmp_path / "wrapped.txt"
        target.write_text("a bird flew over\nthe shining sea beyond the hill\n")
        main(["analyze", str(target), "--reference", str(reference_dir),
              "--order", "2", "--window", "8", "--top-spans", "1"])
        assert "ll.1-2" in capsys.readouterr().out

    def test_spans_can_be_turned_off(self, reference_dir, target, capsys):
        main(["analyze", str(target), "--reference", str(reference_dir),
              "--order", "2", "--top-spans", "0"])
        assert "most variant spans" not in capsys.readouterr().out

    def test_json_carries_the_spans(self, reference_dir, target, capsys):
        main(["analyze", str(target), "--reference", str(reference_dir),
              "--order", "2", "--window", "6", "--top-spans", "2",
              "--report", "json"])
        report = json.loads(capsys.readouterr().out)
        spans = report["documents"][0]["spans"]
        assert spans["window"] == 6
        assert len(spans["most_variant"]) == 2
        assert "first_line" in spans["most_variant"][0]

    def test_a_negative_span_count_exits_2(self, reference_dir, target):
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", str(target), "--reference", str(reference_dir),
                  "--top-spans", "-1"])
        assert excinfo.value.code == 2


@pytest.fixture(scope="module")
def spans():
    """The published run: the moon speech ranked against the inaugurals."""
    reference = markov_cli.read_documents(INAUGURALS)
    target = markov_cli.read_documents([MOON])
    analysis = markov_cli.analyze_documents(
        reference, target, "plain", 3, "loo", window=12, top_spans=5
    )
    return analysis["documents"][0]["spans"]


@requires_corpus
class TestPublishedSpanRanking:
    """The M4 exit criterion: reproduce the prototype's span ranking."""

    def test_the_most_variant_span_is_the_reentry_passage(self, spans):
        top = spans["most_variant"][0]
        assert top["bits_per_token"] == pytest.approx(17.1, abs=0.05)
        assert "25,000 miles per hour" in top["text"]

    def test_the_second_is_the_atlas_passage(self, spans):
        second = spans["most_variant"][1]
        assert second["bits_per_token"] == pytest.approx(16.4, abs=0.05)
        assert "John Glenn" in second["text"]

    def test_the_least_variant_span_is_boilerplate(self, spans):
        bottom = spans["least_variant"][0]
        assert bottom["bits_per_token"] == pytest.approx(4.5, abs=0.05)
        assert "it will be done" in bottom["text"]

    def test_novel_content_outranks_boilerplate_by_about_four_to_one(self, spans):
        """A3: the spread the PRD measured is what makes spans the deliverable
        rather than the document score."""
        top = spans["most_variant"][0]["bits_per_token"]
        bottom = spans["least_variant"][0]["bits_per_token"]
        assert top / bottom == pytest.approx(3.8, abs=0.2)

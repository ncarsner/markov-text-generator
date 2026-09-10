"""Tests for the explanation trace -- what makes a flag checkable."""

import glob
import json
import os

import pytest

import markov_cli
import markov_core
from markov_cli import _evidence, _evidence_lines, explain_span, main

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INAUGURALS = sorted(glob.glob(os.path.join(REPO, "data/input/speech_inaugural_*.txt")))
MOON = os.path.join(REPO, "data/input/speech_we_choose_to_go_to_the_moon.txt")

requires_corpus = pytest.mark.skipif(
    len(INAUGURALS) < 54 or not os.path.exists(MOON),
    reason="public-domain corpus absent; run scripts/fetch_corpus.py",
)

CORPUS = "the cat sat on the mat the cat sat on the log the dog barked".split()


@pytest.fixture
def model():
    return markov_core.NgramModel(CORPUS, order=3)


class TestExplain:
    def test_a_full_context_match_names_the_ngram_and_its_count(self, model):
        explained = model.explain(["the", "cat"], "sat")
        assert explained["backoff_level"] == 2
        assert explained["context_used"] == ["the", "cat"]
        assert explained["count"] == 2

    def test_a_partial_match_reports_the_context_it_could_use(self, model):
        """Backing off is choosing which n-gram to believe; the explanation
        says which one, not merely that it happened. "mat cat sat" never
        occurs, but "cat sat" does, twice."""
        explained = model.explain(["mat", "cat"], "sat")
        assert explained["backoff_level"] == 1
        assert explained["context_used"] == ["cat"]
        assert explained["count"] == 2

    def test_a_word_seen_but_never_in_this_context_backs_off_to_the_word(self, model):
        explained = model.explain(["the", "dog"], "mat")
        assert explained["backoff_level"] == 0
        assert explained["context_used"] == []
        assert explained["count"] == explained["word_count"] == 1

    def test_an_unseen_word_reports_no_evidence_at_all(self, model):
        explained = model.explain(["the", "cat"], "helicopter")
        assert explained["backoff_level"] == -1
        assert explained["count"] == 0
        assert explained["word_count"] == 0

    def test_the_context_offered_is_truncated_to_the_order(self, model):
        """Order 3 can use two words of context; the rest is not evidence."""
        explained = model.explain(["a", "b", "the", "cat"], "sat")
        assert explained["context_offered"] == ["the", "cat"]

    def test_the_explanation_matches_what_the_score_used(self, model):
        for context, word in ((["the", "cat"], "sat"), (["the", "dog"], "mat"),
                              (["the", "cat"], "helicopter")):
            _, level, count = model.score(context, word)
            explained = model.explain(context, word)
            assert explained["backoff_level"] == level
            assert explained["count"] == count


class TestEvidencePhrasing:
    def test_an_unseen_word_says_so(self):
        assert _evidence(
            {"word": "atlas", "backoff_level": -1, "count": 0,
             "context_offered": ["the"], "context_used": []}
        ) == '"atlas" never appears in the reference'

    def test_a_backed_off_word_names_the_context_it_never_followed(self):
        assert _evidence(
            {"word": "causing", "backoff_level": 0, "count": 1,
             "context_offered": ["per", "hour"], "context_used": []}
        ) == '"causing" appears 1x, but never after "per hour"'

    def test_a_matched_ngram_is_quoted_whole(self):
        assert _evidence(
            {"word": "people", "backoff_level": 2, "count": 89,
             "context_offered": ["of", "the"], "context_used": ["of", "the"]}
        ) == '"of the people" appears 89x'

    def test_large_counts_are_readable(self):
        assert "6,722" in _evidence(
            {"word": "of", "backoff_level": 0, "count": 6722,
             "context_offered": ["at"], "context_used": []}
        )


class TestExplainSpan:
    def test_one_entry_per_token_of_the_span(self, model):
        tokens = CORPUS
        scores = model.surprisals(tokens)
        entries = explain_span(model, tokens, scores, 2, 4)
        assert len(entries) == 4
        assert {entry["word"] for entry in entries} == set(tokens[2:6])

    def test_the_most_surprising_token_comes_first(self, model):
        tokens = CORPUS
        scores = model.surprisals(tokens)
        entries = explain_span(model, tokens, scores, 0, 6)
        assert entries == sorted(entries, key=lambda e: -e["bits"])

    def test_each_entry_carries_the_bits_it_contributed(self, model):
        tokens = CORPUS
        scores = model.surprisals(tokens)
        entries = explain_span(model, tokens, scores, 3, 3)
        assert sorted(e["bits"] for e in entries) == sorted(
            round(s, 6) for s in scores[3:6]
        )


class TestEvidenceLines:
    SPAN = {
        "explanation": [
            {"word": "atlas", "backoff_level": -1, "count": 0, "bits": 21.0,
             "context_offered": [], "context_used": []},
            {"word": "the", "backoff_level": 2, "count": 40, "bits": 2.0,
             "context_offered": ["of", "all"], "context_used": ["of", "all"]},
        ]
    }

    def test_a_variant_span_is_explained_by_its_least_expected_word(self):
        line = _evidence_lines(self.SPAN, explain=False, variant=True)[0]
        assert "driven by" in line and "atlas" in line

    def test_a_familiar_span_is_explained_by_its_most_expected_word(self):
        """Quoting a familiar span's worst token would explain the opposite of
        why it is in that list."""
        line = _evidence_lines(self.SPAN, explain=False, variant=False)[0]
        assert "anchored by" in line and "of all the" in line

    def test_the_full_trace_lists_every_token(self):
        lines = _evidence_lines(self.SPAN, explain=True, variant=True)
        assert len(lines) == 2

    def test_the_full_trace_runs_from_the_deciding_token(self):
        variant = _evidence_lines(self.SPAN, explain=True, variant=True)
        familiar = _evidence_lines(self.SPAN, explain=True, variant=False)
        assert "atlas" in variant[0]
        assert "of all the" in familiar[0]

    def test_a_span_without_evidence_renders_nothing(self):
        assert _evidence_lines({"explanation": []}, explain=False, variant=True) == []


class TestReport:
    @pytest.fixture
    def scored(self, tmp_path):
        reference = tmp_path / "ref"
        reference.mkdir()
        (reference / "a.txt").write_text("the cat sat on the mat and the dog sat\n")
        (reference / "b.txt").write_text("the dog ate the rat and the cat ate\n")
        (reference / "c.txt").write_text("the cat sat on the log and the rat sat\n")
        target = tmp_path / "target.txt"
        target.write_text(
            "the cat sat on the mat and the dog sat\n"
            "a helicopter arrived over the shining sea beyond the distant hill\n"
        )
        return reference, target

    def test_every_reported_span_carries_its_evidence_by_default(self, scored, capsys):
        """A4: a flag whose basis is only in the JSON is one most readers will
        never check."""
        reference, target = scored
        main(["analyze", str(target), "--reference", str(reference),
              "--order", "2", "--window", "5", "--top-spans", "2"])
        out = capsys.readouterr().out
        assert out.count("driven by") == 2
        assert out.count("anchored by") == 2

    def test_the_trace_names_a_count_in_the_reference(self, scored, capsys):
        reference, target = scored
        main(["analyze", str(target), "--reference", str(reference),
              "--order", "2", "--window", "5", "--top-spans", "1"])
        out = capsys.readouterr().out
        assert "never appears in the reference" in out or "appears" in out

    def test_explain_expands_the_trace_to_every_token(self, scored, capsys):
        reference, target = scored
        main(["analyze", str(target), "--reference", str(reference),
              "--order", "2", "--window", "5", "--top-spans", "1", "--explain"])
        out = capsys.readouterr().out
        assert "driven by" not in out
        # Five tokens per span, at both ends of the ranking.
        assert len([l for l in out.splitlines() if "appears" in l or "never" in l]) == 10

    def test_json_carries_the_whole_trace(self, scored, capsys):
        reference, target = scored
        main(["analyze", str(target), "--reference", str(reference),
              "--order", "2", "--window", "5", "--top-spans", "1",
              "--report", "json"])
        report = json.loads(capsys.readouterr().out)
        explanation = report["documents"][0]["spans"]["most_variant"][0]["explanation"]
        assert len(explanation) == 5
        assert {"word", "backoff_level", "count", "word_count", "bits"} <= set(
            explanation[0]
        )


@pytest.fixture(scope="module")
def published_spans():
    reference = markov_cli.read_documents(INAUGURALS)
    target = markov_cli.read_documents([MOON])
    analysis = markov_cli.analyze_documents(
        reference, target, "plain", 3, "loo", window=12, top_spans=5
    )
    return analysis["documents"][0]["spans"]


@requires_corpus
class TestTraceability:
    """A4: every flagged span reports the reference count behind it."""

    def test_every_span_at_both_ends_carries_a_full_trace(self, published_spans):
        for end in ("most_variant", "least_variant"):
            for span in published_spans[end]:
                assert len(span["explanation"]) == 12
                assert all("count" in entry for entry in span["explanation"])

    def test_the_top_span_is_driven_by_a_word_absent_from_the_reference(
        self, published_spans
    ):
        driver = published_spans["most_variant"][0]["explanation"][0]
        assert driver["backoff_level"] == -1
        assert driver["count"] == 0

    def test_a_reported_count_matches_the_corpus(self, published_spans):
        """The trace is only worth anything if the numbers are real: rebuild the
        count independently and compare."""
        reference_tokens = []
        for path in INAUGURALS:
            with open(path, encoding="utf-8") as handle:
                reference_tokens.extend(markov_core.plain_tokens(handle.read()))

        for span in published_spans["least_variant"]:
            for entry in span["explanation"]:
                if entry["backoff_level"] < 1:
                    continue
                ngram = entry["context_used"] + [entry["word"]]
                occurrences = sum(
                    reference_tokens[i:i + len(ngram)] == ngram
                    for i in range(len(reference_tokens) - len(ngram) + 1)
                )
                assert occurrences == entry["count"], ngram
                return

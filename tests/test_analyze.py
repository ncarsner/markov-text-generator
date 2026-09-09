"""Tests for document scoring and calibration -- the principal application."""

import glob
import json
import os
import statistics

import pytest

import markov_cli
import markov_core
from markov_cli import analyze_documents, calibrate, deviation, main

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INAUGURALS = sorted(glob.glob(os.path.join(REPO, "data/input/speech_inaugural_*.txt")))
MOON = os.path.join(REPO, "data/input/speech_we_choose_to_go_to_the_moon.txt")
LINCOLN = os.path.join(
    REPO, "data/input/speech_inaugural_abraham_lincoln_second_inaugural_address.txt"
)

requires_corpus = pytest.mark.skipif(
    len(INAUGURALS) < 54 or not os.path.exists(MOON),
    reason="public-domain corpus absent; run scripts/fetch_corpus.py",
)

# Overlapping enough that leave-one-out scores differ without any document
# being unreachable, which a corpus of identical documents would be.
REFERENCE = [
    ("a.txt", "the cat sat on the mat and the dog sat on the log\n"),
    ("b.txt", "the dog ate the rat and the cat ate the mat\n"),
    ("c.txt", "the cat sat on the log and the rat sat on the mat\n"),
]
OUTSIDER = ("outsider.txt", "a bird flew over the shining sea beyond the hill\n")


def tokenized(documents):
    return [markov_core.plain_tokens(text) for _, text in documents]


def ten_documents():
    """Ten distinguishable documents, the minimum a holdout baseline accepts."""
    return [
        (f"d{i}.txt", f"the cat sat on the mat number {i} and the dog sat there\n")
        for i in range(10)
    ]


class TestCalibration:
    def test_leave_one_out_scores_every_document(self):
        calibration = calibrate(tokenized(REFERENCE), 2, "loo")
        assert calibration["documents"] == 3
        assert calibration["method"] == "leave-one-out"

    def test_a_single_document_cannot_be_calibrated(self):
        """One document has nothing to be compared against."""
        with pytest.raises(SystemExit) as excinfo:
            calibrate(tokenized(REFERENCE[:1]), 2, "loo")
        assert "at least 2" in str(excinfo.value)

    def test_spread_is_the_sample_standard_deviation(self):
        """The reference is a sample of the language being characterized, not
        the whole of it, so the n-1 denominator is the correct one."""
        expected = []
        for held in range(len(REFERENCE)):
            others = [t for i, t in enumerate(tokenized(REFERENCE)) if i != held]
            model = markov_core.NgramModel(
                [tok for doc in others for tok in doc], order=2
            )
            expected.append(model.bits_per_token(tokenized(REFERENCE)[held]))

        calibration = calibrate(tokenized(REFERENCE), 2, "loo")
        assert calibration["surprisal"]["stdev"] == pytest.approx(
            statistics.stdev(expected)
        )
        assert calibration["surprisal"]["stdev"] != pytest.approx(
            statistics.pstdev(expected)
        )

    def test_calibration_covers_novel_ngrams_too(self):
        """F5 reports the novel rate against its own expected range, so the
        range has to be measured in the same pass."""
        calibration = calibrate(tokenized(REFERENCE), 2, "loo")
        assert 0 <= calibration["novel_ngram_rate"]["mean"] <= 1
        assert calibration["novel_ngram_rate"]["stdev"] > 0

    def test_holdout_needs_enough_documents_and_says_so(self):
        with pytest.raises(SystemExit) as excinfo:
            calibrate(tokenized(REFERENCE), 2, "holdout")
        assert "loo" in str(excinfo.value)

    def test_holdout_scores_every_fifth_document(self):
        calibration = calibrate(tokenized(ten_documents()), 2, "holdout")
        assert calibration["documents"] == 2
        assert calibration["method"] == "holdout"

    def test_holdout_does_not_train_on_what_it_scores(self):
        """The held-out documents must be absent from the model that scores
        them. Left in, the corpus is grading its own homework and the expected
        range comes out too narrow and too low."""
        documents = tokenized(ten_documents())
        held_out = [0, 5]
        trained_on = [t for i, t in enumerate(documents) if i not in held_out]
        model = markov_core.NgramModel(
            [tok for doc in trained_on for tok in doc], order=2
        )
        expected = [model.bits_per_token(documents[i]) for i in held_out]

        calibration = calibrate(documents, 2, "holdout")
        assert calibration["surprisal"]["mean"] == pytest.approx(
            statistics.fmean(expected)
        )

    def test_holdout_reports_how_many_documents_backed_the_model(self):
        calibration = calibrate(tokenized(ten_documents()), 2, "holdout")
        assert calibration["trained_on"] == 8

    def test_leave_one_out_trains_on_everything_but_one(self):
        calibration = calibrate(tokenized(REFERENCE), 2, "loo")
        assert calibration["trained_on"] == 2

    def test_the_two_baselines_are_different_estimators(self):
        documents = tokenized(ten_documents())
        assert (
            calibrate(documents, 2, "loo")["surprisal"]["mean"]
            != calibrate(documents, 2, "holdout")["surprisal"]["mean"]
        )


class TestDeviation:
    def test_z_counts_standard_deviations_from_the_mean(self):
        assert deviation(10.0, {"mean": 9.0, "stdev": 0.5}) == pytest.approx(2.0)
        assert deviation(8.5, {"mean": 9.0, "stdev": 0.5}) == pytest.approx(-1.0)

    def test_no_spread_is_refused_rather_than_dividing_by_zero(self):
        with pytest.raises(SystemExit) as excinfo:
            deviation(10.0, {"mean": 9.0, "stdev": 0.0})
        assert "spread" in str(excinfo.value)


class TestScoring:
    def test_an_outside_document_is_scored_against_the_whole_reference(self):
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        document = analysis["documents"][0]

        whole = markov_core.NgramModel(
            [tok for doc in tokenized(REFERENCE) for tok in doc], order=2
        )
        assert document["held_out"] is False
        assert document["bits_per_token"] == pytest.approx(
            whole.bits_per_token(markov_core.plain_tokens(OUTSIDER[1])), abs=1e-6
        )

    def test_a_reference_document_is_scored_without_itself(self):
        """Left in the model, a document scores itself and the corpus flatters
        it; the number would say more about leakage than about the language."""
        analysis = analyze_documents(REFERENCE, [REFERENCE[0]], "plain", 2, "loo")
        document = analysis["documents"][0]

        whole = markov_core.NgramModel(
            [tok for doc in tokenized(REFERENCE) for tok in doc], order=2
        )
        included = whole.bits_per_token(markov_core.plain_tokens(REFERENCE[0][1]))

        assert document["held_out"] is True
        assert document["bits_per_token"] > included

    def test_an_unlike_document_is_called_variant(self):
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        assert analysis["documents"][0]["verdict"] == "variant"
        assert analysis["documents"][0]["z"] > markov_cli.VARIANT_Z

    def test_a_typical_document_is_not_called_variant(self):
        analysis = analyze_documents(REFERENCE, [REFERENCE[0]], "plain", 2, "loo")
        assert analysis["documents"][0]["verdict"] == "typical"
        assert abs(analysis["documents"][0]["z"]) <= markov_cli.VARIANT_Z

    def test_the_verdict_follows_the_threshold(self, monkeypatch):
        monkeypatch.setattr(markov_cli, "VARIANT_Z", 0.0)
        analysis = analyze_documents(REFERENCE, [REFERENCE[0]], "plain", 2, "loo")
        assert analysis["documents"][0]["verdict"] == "variant"

    def test_every_target_is_scored(self):
        analysis = analyze_documents(
            REFERENCE, [OUTSIDER, REFERENCE[1]], "plain", 2, "loo"
        )
        assert [d["path"] for d in analysis["documents"]] == ["outsider.txt", "b.txt"]

    def test_order_changes_the_score(self):
        first = analyze_documents(REFERENCE, [OUTSIDER], "plain", 1, "loo")
        second = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        assert first["order"] == 1
        assert (
            first["documents"][0]["bits_per_token"]
            != second["documents"][0]["bits_per_token"]
        )

    def test_tokenizer_choice_is_honored(self):
        mixed = [("m.txt", "The cat sat. the cat sat\n")] + REFERENCE
        plain = analyze_documents(mixed, [OUTSIDER], "plain", 2, "loo")
        whitespace = analyze_documents(mixed, [OUTSIDER], "whitespace", 2, "loo")
        assert (
            plain["documents"][0]["bits_per_token"]
            != whitespace["documents"][0]["bits_per_token"]
        )

    def test_a_target_shorter_than_the_order_is_refused(self):
        with pytest.raises(SystemExit) as excinfo:
            analyze_documents(REFERENCE, [("tiny.txt", "a b\n")], "plain", 3, "loo")
        assert "too short" in str(excinfo.value)

    def test_small_reference_is_flagged(self):
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        assert any("sparsity" in w for w in analysis["warnings"])

    def test_a_document_is_ranked_against_the_reference(self):
        """A count of documents out-scored needs no statistics to read, which
        is the point of reporting it alongside z."""
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        assert analysis["documents"][0]["reference_documents_below"] == 3

        typical = analyze_documents(REFERENCE, [REFERENCE[0]], "plain", 2, "loo")
        assert typical["documents"][0]["reference_documents_below"] < 3

    def test_the_ranking_counts_documents_scoring_lower(self):
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        document = analysis["documents"][0]
        calibration = calibrate(tokenized(REFERENCE), 2, "loo")
        assert document["reference_documents_below"] == sum(
            score < document["bits_per_token"] for score in calibration["scores"]
        )

    def test_reference_composition_accompanies_the_scores(self):
        """'Expected' means 'typical of what you supplied', so what was
        supplied is reported with every score."""
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        assert analysis["reference"]["documents"] == 3
        assert analysis["reference"]["tokens"] == sum(
            len(t) for t in tokenized(REFERENCE)
        )


class TestTextReport:
    def test_states_the_score_the_range_and_the_verdict(self, tmp_path, capsys):
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)
        target = tmp_path / "outsider.txt"
        target.write_text(OUTSIDER[1])

        assert main(
            ["analyze", str(target), "--reference", str(reference), "--order", "2"]
        ) == 0
        out = capsys.readouterr().out

        assert "bits/token" in out
        assert "+/-" in out
        assert "VARIANT" in out
        assert "scoring each of the 3 reference documents" in out

    def test_explains_where_the_expected_range_came_from(self, tmp_path, capsys):
        """The report is read by people deciding whether to act on a flag, so
        it states the procedure rather than naming the technique."""
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)

        main(["analyze", str(reference / "a.txt"), "--reference", str(reference),
              "--order", "2"])
        out = capsys.readouterr().out
        assert "scoring each of the 3 reference documents against the other 2" in out
        assert "leave-one-out" not in out

    def test_ranks_the_document_against_the_reference_in_plain_words(
        self, tmp_path, capsys
    ):
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)
        target = tmp_path / "outsider.txt"
        target.write_text(OUTSIDER[1])

        main(["analyze", str(target), "--reference", str(reference), "--order", "2"])
        assert "more unusual than 3 of the 3 reference documents" in capsys.readouterr().out

    def test_says_what_variant_does_not_mean(self, tmp_path, capsys):
        """The PRD requires this in the output, not only in the docs."""
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)

        main(["analyze", str(reference / "a.txt"), "--reference", str(reference),
              "--order", "2"])
        out = capsys.readouterr().out
        assert "does not mean wrong" in out
        assert "typical of the corpus you supplied" in out

    def test_flags_a_target_that_came_from_the_reference(self, tmp_path, capsys):
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)

        main(["analyze", str(reference / "a.txt"), "--reference", str(reference),
              "--order", "2"])
        assert "built without it" in capsys.readouterr().out


class TestJsonReport:
    @pytest.fixture
    def reference_dir(self, tmp_path):
        reference = tmp_path / "ref"
        reference.mkdir()
        for name, text in REFERENCE:
            (reference / name).write_text(text)
        return reference

    def test_carries_score_calibration_and_composition(self, reference_dir, capsys):
        main(["analyze", str(reference_dir / "a.txt"), "--reference",
              str(reference_dir), "--order", "2", "--report", "json"])
        report = json.loads(capsys.readouterr().out)

        assert report["calibration"]["method"] == "leave-one-out"
        assert report["reference"]["documents"] == 3
        assert report["documents"][0]["verdict"] in ("typical", "variant")

    def test_output_is_identical_across_runs(self, reference_dir, capsys):
        argv = ["analyze", str(reference_dir / "a.txt"), "--reference",
                str(reference_dir), "--order", "2", "--report", "json"]
        main(argv)
        first = capsys.readouterr().out
        main(argv)
        assert capsys.readouterr().out == first


class TestOptionValidation:
    def test_analyze_requires_a_reference(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", "target.txt"])
        assert excinfo.value.code == 2

    def test_analyze_requires_a_target(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", "--reference", str(tmp_path)])
        assert excinfo.value.code == 2

    def test_unknown_baseline_exits_2(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", "t.txt", "--reference", str(tmp_path),
                  "--baseline", "bootstrap"])
        assert excinfo.value.code == 2


@pytest.fixture(scope="module")
def analysis():
    """The published run: 54 inaugurals at order 3, leave-one-out calibrated."""
    reference = markov_cli.read_documents(INAUGURALS)
    targets = markov_cli.read_documents([MOON, LINCOLN])
    return analyze_documents(reference, targets, "plain", 3, "loo")


@requires_corpus
class TestPublishedBaseline:
    """The M3 exit criterion: reproduce the PRD's measured figures exactly."""

    def test_expected_range_is_9_01_plus_or_minus_0_47(self, analysis):
        assert analysis["calibration"]["surprisal_mean"] == pytest.approx(9.01, abs=5e-3)
        assert analysis["calibration"]["surprisal_stdev"] == pytest.approx(0.47, abs=5e-3)

    def test_an_out_of_domain_document_scores_beyond_the_reference(self, analysis):
        moon = analysis["documents"][0]
        assert moon["bits_per_token"] == pytest.approx(10.36, abs=5e-3)
        assert moon["z"] == pytest.approx(2.85, abs=5e-3)
        assert moon["novel_ngram_rate"] == pytest.approx(0.86, abs=5e-3)
        assert moon["verdict"] == "variant"

    def test_a_held_out_reference_document_stays_inside_it(self, analysis):
        """A2 and A1: out-of-domain clears +2.5, in-domain does not."""
        lincoln = analysis["documents"][1]
        assert lincoln["held_out"] is True
        assert lincoln["bits_per_token"] == pytest.approx(9.99, abs=5e-3)
        assert lincoln["z"] == pytest.approx(2.06, abs=5e-3)
        assert lincoln["verdict"] == "typical"

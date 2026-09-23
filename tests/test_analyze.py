"""Tests for document scoring and calibration -- the principal application."""

import glob
import json
import os
import random
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

    def test_leave_one_out_by_subtraction_equals_leave_one_out_by_rebuilding(self):
        """The speed of calibration rests on this: subtracting a document's
        counts from the whole corpus is the same model as never adding them,
        seam n-grams included. Rebuilt here independently of the engine, so a
        change to the subtraction has something to disagree with."""
        documents = tokenized(REFERENCE)
        expected = []
        for held, tokens in enumerate(documents):
            model = markov_core.NgramModel(
                [tok for i, doc in enumerate(documents) if i != held for tok in doc],
                order=2,
            )
            expected.append(markov_cli.measure(model, tokens))

        calibration = calibrate(documents, 2, "loo")
        for key, field in (
            ("bits_per_token", "surprisal"),
            ("novel_ngram_rate", "novel_ngram_rate"),
        ):
            values = [score[key] for score in expected]
            assert calibration[field]["mean"] == pytest.approx(
                statistics.fmean(values)
            )
            assert calibration[field]["stdev"] == pytest.approx(
                statistics.stdev(values)
            )

    def test_calibration_leaves_the_model_it_borrowed_intact(self):
        """The same model goes on to score the targets, so calibration must
        hand it back exactly as it found it."""
        documents = tokenized(REFERENCE)
        model = markov_core.NgramModel.from_documents(documents, order=2)
        before = markov_core.NgramModel.from_documents(documents, order=2)

        calibrate(documents, 2, "loo", model=model)

        assert model.counts == before.counts
        assert (model.total, model.vocab) == (before.total, before.vocab)

    def test_a_caller_s_model_and_a_built_one_calibrate_alike(self):
        documents = tokenized(REFERENCE)
        supplied = calibrate(
            documents,
            2,
            "loo",
            model=markov_core.NgramModel.from_documents(documents, order=2),
        )
        assert supplied == calibrate(documents, 2, "loo")

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


def varied_documents(count, repeats=1):
    """Documents that overlap enough to be scorable and differ enough to spread.

    Identical documents score identically, which leaves no spread and no z at
    all. Each of these carries a different amount of the shared phrase, so
    leave-one-out produces a real distribution. ``repeats`` scales them up when
    a test needs a corpus past MIN_REFERENCE_TOKENS.
    """
    shared = "the cat sat on the mat and the dog ate the rat"
    return [
        (f"d{i}.txt", " ".join([shared] * ((i + 1) * repeats)
                               + [f"unique{i}word{j}" for j in range(5)]) + "\n")
        for i in range(count)
    ]


class TestVerdictCeiling:
    """What a reference of a given size can say about its own members."""

    @pytest.mark.parametrize(
        "documents, ceiling",
        [(2, 0.707107), (3, 1.154701), (5, 1.788854), (8, 2.474874),
         (9, 2.666667), (54, 7.212386)],
    )
    def test_the_bound_is_n_minus_one_over_root_n(self, documents, ceiling):
        assert markov_cli.verdict_ceiling(documents) == pytest.approx(ceiling)

    def test_a_flat_threshold_was_unreachable_below_nine_documents(self):
        """Why the threshold is derived from this bound rather than fixed at
        2.5: below 9 documents the bound sits under 2.5, so "typical" reported
        the size of the corpus and not the text."""
        assert markov_cli.verdict_ceiling(8) < markov_cli.VARIANT_Z
        assert markov_cli.verdict_ceiling(9) > markov_cli.VARIANT_Z
        assert markov_cli.verdict_threshold(8, held_out=True) < (
            markov_cli.verdict_ceiling(8)
        )

    @pytest.mark.parametrize("count", [3, 5, 9, 20])
    def test_no_reference_document_ever_exceeds_it(self, count):
        """The bound is arithmetic, so it holds for any corpus. Checked against
        real leave-one-out scores rather than asserted from the formula."""
        calibration = calibrate(tokenized(varied_documents(count)), 2, "loo")
        ceiling = markov_cli.verdict_ceiling(calibration["documents"])
        for score in calibration["scores"]:
            assert abs(deviation(score, calibration["surprisal"])) <= ceiling + 1e-9

    def test_it_is_reported_with_the_calibration(self):
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        assert analysis["calibration"]["ceiling"] == pytest.approx(
            markov_cli.verdict_ceiling(3)
        )

    def test_a_corpus_with_no_room_to_work_in_warns(self):
        """Three documents leave the threshold on top of the ceiling: variant
        is reachable in principle and out of reach in practice."""
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        warning = next(w for w in analysis["warnings"] if "of a possible" in w)
        assert "1.15 z of a possible 1.15" in warning

    def test_a_corpus_large_enough_does_not_warn(self):
        documents = varied_documents(10)
        analysis = analyze_documents(documents, [OUTSIDER], "plain", 2, "loo")
        assert markov_cli.verdict_threshold(10, held_out=True) < (
            markov_cli.CROWDED_CEILING * markov_cli.verdict_ceiling(10)
        )
        assert not any("of a possible" in w for w in analysis["warnings"])

    def test_the_two_size_warnings_are_independent(self):
        """A corpus can be long enough in words and still hold too few
        documents: the word gate does not cover for the document gate."""
        analysis = analyze_documents(
            varied_documents(3, repeats=2_900), [OUTSIDER], "plain", 2, "loo"
        )
        assert analysis["reference"]["tokens"] > markov_cli.MIN_REFERENCE_TOKENS
        assert not any("sparsity" in w for w in analysis["warnings"])
        assert any("of a possible" in w for w in analysis["warnings"])

    def test_the_bound_follows_the_scores_not_the_corpus(self):
        """Holdout scores every fifth document, so ten documents produce two
        scores and a ceiling of 0.71 -- far tighter than the corpus size
        suggests, and tight enough that variant is out of reach."""
        documents = varied_documents(10)
        analysis = analyze_documents(
            documents, [OUTSIDER], "plain", 2, "holdout"
        )
        assert analysis["reference"]["documents"] == 10
        assert analysis["calibration"]["documents"] == 2
        assert analysis["calibration"]["ceiling"] == pytest.approx(
            markov_cli.verdict_ceiling(2)
        )
        assert any("of a possible" in w for w in analysis["warnings"])

    def test_the_warning_boundary_is_where_the_threshold_crowds_the_ceiling(self):
        """Six documents leave the threshold at 91% of the ceiling and seven at
        86%, so the warning turns off between them."""
        assert markov_cli.verdict_threshold(6, held_out=True) >= (
            markov_cli.CROWDED_CEILING * markov_cli.verdict_ceiling(6)
        )
        assert markov_cli.verdict_threshold(7, held_out=True) < (
            markov_cli.CROWDED_CEILING * markov_cli.verdict_ceiling(7)
        )

    def test_a_threshold_exactly_at_the_limit_still_warns(self, monkeypatch):
        """The limit is inclusive. Pinned by moving it onto a ratio a real
        corpus produces, since no document count lands exactly on 90%."""
        monkeypatch.setattr(
            markov_cli,
            "CROWDED_CEILING",
            markov_cli.verdict_threshold(7, held_out=True)
            / markov_cli.verdict_ceiling(7),
        )
        analysis = analyze_documents(
            varied_documents(7), [OUTSIDER], "plain", 2, "loo"
        )
        assert any("of a possible" in w for w in analysis["warnings"])

    def test_the_report_states_the_bound(self):
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        report = markov_cli.format_analysis(analysis, ["ref"])
        assert "which can reach at most 1.15" in report
        assert "which is not bounded" in report

    def test_a_held_out_document_is_told_what_bounds_it(self):
        analysis = analyze_documents(REFERENCE, [REFERENCE[0]], "plain", 2, "loo")
        report = markov_cli.format_analysis(analysis, ["ref"])
        assert "bounded at 1.15 by the 3 documents" in report

    def test_an_outside_target_is_not_told_it_is_bounded(self):
        """It is not part of the set it is measured against, so it is not."""
        analysis = analyze_documents(REFERENCE, [OUTSIDER], "plain", 2, "loo")
        report = markov_cli.format_analysis(analysis, ["ref"])
        assert "bounded at" not in report



class TestVerdictThreshold:
    """How far a document must deviate to be called variant, given how many
    reference documents set the expectation."""

    @pytest.mark.parametrize(
        "df, published",
        [(1, 12.706), (5, 2.571), (10, 2.228), (30, 2.042), (1000, 1.962)],
    )
    def test_t_values_match_the_published_table(self, df, published):
        """The standard library stops at the normal distribution, so the t
        quantile is computed here. Checked against a printed table rather than
        against the implementation that produced it."""
        assert markov_cli._t_critical(0.05, df) == pytest.approx(published, abs=0.002)

    @pytest.mark.parametrize("held_out", [True, False])
    def test_it_converges_to_the_flat_threshold(self, held_out):
        """A reference large enough to pin down its own range reads as it
        always did: the size adjustment is the correction for not having one."""
        assert markov_cli.verdict_threshold(100_000, held_out) == pytest.approx(
            markov_cli.VARIANT_Z, abs=0.01
        )

    @pytest.mark.parametrize("documents", [3, 4, 5, 8, 9, 20, 54])
    def test_a_member_is_held_inside_its_ceiling(self, documents):
        """The failure the flat threshold had: asking for more than the
        arithmetic allows. This one cannot, at any size."""
        assert markov_cli.verdict_threshold(documents, held_out=True) < (
            markov_cli.verdict_ceiling(documents)
        )

    def test_two_scores_can_flag_nothing(self):
        """Two documents sit at +/- 0.707 by construction. There is no spread
        to have an opinion about, so the threshold is the unreachable bound."""
        assert markov_cli.verdict_threshold(2, held_out=True) == pytest.approx(
            markov_cli.verdict_ceiling(2)
        )

    @pytest.mark.parametrize("documents", [3, 5, 9, 20, 54])
    def test_the_two_kinds_straddle_the_flat_threshold(self, documents):
        """A flat 2.5 was too strict for a member and too loose for an outside
        target, at the same time and for the same reason."""
        assert markov_cli.verdict_threshold(documents, held_out=True) < (
            markov_cli.VARIANT_Z
        )
        assert markov_cli.verdict_threshold(documents, held_out=False) > (
            markov_cli.VARIANT_Z
        )

    def test_it_moves_in_opposite_directions_as_the_reference_grows(self):
        """A member is squeezed by the ceiling, which lifts as documents are
        added; an outside target is squeezed by a noisy spread, which settles."""
        members = [markov_cli.verdict_threshold(n, True) for n in range(3, 60)]
        outside = [markov_cli.verdict_threshold(n, False) for n in range(3, 60)]
        assert members == sorted(members)
        assert outside == sorted(outside, reverse=True)

    @pytest.mark.parametrize("documents", [5, 20])
    @pytest.mark.parametrize("held_out", [True, False])
    def test_the_false_flag_rate_holds_at_every_size(self, documents, held_out):
        """What the threshold is for. Scores drawn from one distribution, so
        nothing is genuinely variant; the share called variant anyway should be
        variant_alpha() whatever the reference size, which a flat 2.5 is not:
        it flags 0% of members at 5 documents and would flag 1.24% only if the
        expected range were known exactly."""
        rng = random.Random(20260920)
        threshold = markov_cli.verdict_threshold(documents, held_out)
        trials, flagged = 20_000, 0
        for _ in range(trials):
            scores = [rng.gauss(0, 1) for _ in range(documents)]
            mean = statistics.fmean(scores)
            spread = statistics.stdev(scores, mean)
            value = scores[0] if held_out else rng.gauss(0, 1)
            flagged += abs((value - mean) / spread) > threshold
        # Four standard errors of the sampling noise at this many trials.
        assert flagged / trials == pytest.approx(
            markov_cli.variant_alpha(), abs=0.0045
        )

    def test_it_follows_variant_z(self, monkeypatch):
        """VARIANT_Z stays the one number that sets how readily the tool calls
        something variant; the size adjustment only says what it means here."""
        relaxed = markov_cli.verdict_threshold(20, held_out=True)
        monkeypatch.setattr(markov_cli, "VARIANT_Z", 3.5)
        assert markov_cli.verdict_threshold(20, held_out=True) > relaxed

    def test_a_member_and_an_outside_target_are_held_to_different_numbers(self):
        documents = varied_documents(9)
        analysis = analyze_documents(
            documents, [documents[0], OUTSIDER], "plain", 2, "loo"
        )
        member, outsider = analysis["documents"]
        assert member["threshold"] == pytest.approx(
            markov_cli.verdict_threshold(9, held_out=True)
        )
        assert outsider["threshold"] == pytest.approx(
            markov_cli.verdict_threshold(9, held_out=False)
        )
        assert member["threshold"] < outsider["threshold"]

    def test_a_small_reference_can_now_flag_its_own_member(self):
        """The point of the change. Eight documents bound a member at 2.47, so
        a flat 2.5 called this document typical however far out it sat; it
        clears the size-adjusted 2.03 instead."""
        documents = varied_documents(8)
        analysis = analyze_documents(documents, [documents[0]], "plain", 2, "loo")
        document = analysis["documents"][0]
        assert document["verdict"] == "variant"
        assert document["threshold"] < abs(document["z"]) < markov_cli.VARIANT_Z

    def test_the_verdict_uses_the_threshold_it_reports(self):
        documents = varied_documents(9)
        analysis = analyze_documents(
            documents, documents + [OUTSIDER], "plain", 2, "loo"
        )
        for document in analysis["documents"]:
            expected = (
                "variant" if abs(document["z"]) > document["threshold"] else "typical"
            )
            assert document["verdict"] == expected

    def test_the_report_states_both_thresholds(self):
        analysis = analyze_documents(varied_documents(9), [OUTSIDER], "plain", 2, "loo")
        report = markov_cli.format_analysis(analysis, ["ref"])
        assert "variant past 2.09 z for a reference document" in report
        assert "past 3.38 z for a target from outside" in report

    def test_a_held_out_document_is_told_what_it_had_to_clear(self):
        documents = varied_documents(9)
        analysis = analyze_documents(documents, [documents[0]], "plain", 2, "loo")
        report = markov_cli.format_analysis(analysis, ["ref"])
        assert "called variant past 2.09 z and bounded at 2.67" in report


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
        analysis = analyze_documents(varied_documents(9), [OUTSIDER], "plain", 2, "loo")
        document = analysis["documents"][0]
        assert document["verdict"] == "variant"
        assert document["z"] > document["threshold"]

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
        for name, text in varied_documents(9):
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
        assert "scoring each of the 9 reference documents" in out

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

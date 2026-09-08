"""Tests for the command-line layer: dispatch, reference resolution, reporting."""

import json
import re
import subprocess
import sys

import pytest

import markov_cli
from markov_cli import corpus_report, main, resolve_reference

from conftest import CORPUS_LINES

# "      2        1.75     78%" -- a table row, as distinct from the composition
# line above it, which also begins with a digit.
TABLE_ROW = re.compile(r"^\s+\d+\s+\d+\.\d\d\s+\d+%$")


def table_rows(output):
    return [line for line in output.splitlines() if TABLE_ROW.match(line)]


@pytest.fixture
def corpus_dir(tmp_path):
    """A directory of three small documents, plus one non-.txt distractor."""
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "a.txt").write_text("the cat sat on the mat\n")
    (directory / "b.txt").write_text("the cat ate the rat\n")
    (directory / "c.txt").write_text("the dog sat on the log\n")
    (directory / "notes.md").write_text("not part of the corpus\n")
    return directory


class TestDispatch:
    def test_no_subcommand_exits_2(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
        assert "stats" in capsys.readouterr().err

    def test_unknown_subcommand_exits_2(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["analyse"])
        assert excinfo.value.code == 2

    def test_stats_requires_a_reference(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["stats"])
        assert excinfo.value.code == 2

    def test_falls_back_to_sys_argv(self, corpus_dir, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["markov", "stats", "--reference", str(corpus_dir)]
        )
        assert main() == 0
        assert "3 documents" in capsys.readouterr().out


class TestReferenceResolution:
    def test_directory_takes_only_txt_files(self, corpus_dir):
        resolved = resolve_reference([str(corpus_dir)])
        assert [p.rsplit("/", 1)[-1] for p in resolved] == ["a.txt", "b.txt", "c.txt"]

    def test_glob_is_expanded_when_the_shell_did_not(self, corpus_dir):
        """A quoted pattern must behave like an unquoted one -- the README
        recommends quoting to keep a stray *.txt out of published figures."""
        resolved = resolve_reference([str(corpus_dir / "[ab].txt")])
        assert [p.rsplit("/", 1)[-1] for p in resolved] == ["a.txt", "b.txt"]

    def test_explicit_files_keep_their_given_order(self, corpus_dir):
        given = [str(corpus_dir / "c.txt"), str(corpus_dir / "a.txt")]
        assert resolve_reference(given) == given

    def test_overlapping_patterns_do_not_double_count(self, corpus_dir):
        """The same document reached two ways must contribute once."""
        resolved = resolve_reference([str(corpus_dir), str(corpus_dir / "a.txt")])
        assert len(resolved) == 3

    def test_missing_file_is_named(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            resolve_reference([str(tmp_path / "nope.txt")])
        assert "nope.txt" in str(excinfo.value)

    def test_pattern_matching_nothing_is_named(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            resolve_reference([str(tmp_path / "absent_*.txt")])
        assert "absent_*.txt" in str(excinfo.value)

    def test_directory_without_txt_files_is_named(self, tmp_path):
        (tmp_path / "notes.md").write_text("nothing to see\n")
        with pytest.raises(SystemExit) as excinfo:
            resolve_reference([str(tmp_path)])
        assert str(tmp_path) in str(excinfo.value)

    def test_reads_utf8_content(self, tmp_path):
        source = tmp_path / "smart.txt"
        source.write_text("the “new” order — begins\n", encoding="utf-8")
        report = corpus_report(
            markov_cli.read_documents([str(source)]), "whitespace", 1
        )
        assert report["tokens"] == 5

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="EncodingWarning added in 3.10"
    )
    def test_encoding_is_never_left_to_the_locale(self, tmp_path):
        """Reading UTF-8 content is not evidence on a UTF-8 machine -- it passes
        just as well with the encoding omitted. Re-run under
        -X warn_default_encoding, where an implicit open() is an error, so this
        test fails on the machine where the bug would actually bite."""
        source = tmp_path / "smart.txt"
        source.write_text("the “new” order — begins\n", encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "-X", "warn_default_encoding",
                "-W", "error::EncodingWarning",
                "markov_cli.py", "stats",
                "--reference", str(source),
                "--order", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestCorpusReport:
    def test_counts_tokens_and_vocabulary(self, corpus_dir):
        report = corpus_report(
            markov_cli.read_documents(resolve_reference([str(corpus_dir)])), "plain", 1
        )
        assert report["tokens"] == 17
        assert report["vocabulary"] == 9

    def test_per_document_contribution_is_reported(self, corpus_dir):
        report = corpus_report(
            markov_cli.read_documents(resolve_reference([str(corpus_dir)])), "plain", 1
        )
        assert [d["tokens"] for d in report["documents"]] == [6, 5, 6]
        assert sum(d["tokens"] for d in report["documents"]) == report["tokens"]

    def test_reports_every_order_up_to_the_maximum(self, corpus_dir):
        report = corpus_report(
            markov_cli.read_documents(resolve_reference([str(corpus_dir)])), "plain", 3
        )
        assert [row["order"] for row in report["orders"]] == [1, 2, 3]

    def test_branching_matches_the_engine(self, corpus_dir):
        """The CLI must report what markov_core measures, not its own count."""
        import markov_core

        documents = markov_cli.read_documents(resolve_reference([str(corpus_dir)]))
        tokens = markov_core.plain_tokens(" ".join(t for _, t in documents))
        expected, forced = markov_core.chain_statistics(tokens, 2)

        report = corpus_report(documents, "plain", 2)
        row = report["orders"][1]
        assert row["branching"] == pytest.approx(expected)
        assert row["forced"] == pytest.approx(forced)

    def test_tokenizer_choice_changes_the_numbers(self, tmp_path):
        """plain and whitespace are two views of one corpus; the report says
        which it used, because the figures are not comparable across them."""
        source = tmp_path / "mixed.txt"
        source.write_text("Freedom! is not free. freedom is not free\n")
        documents = markov_cli.read_documents([str(source)])

        plain = corpus_report(documents, "plain", 1)
        whitespace = corpus_report(documents, "whitespace", 1)

        assert plain["vocabulary"] < whitespace["vocabulary"]
        assert plain["tokenizer"] == "plain"
        assert whitespace["tokenizer"] == "whitespace"

    def test_small_corpus_is_flagged(self, corpus_dir):
        report = corpus_report(
            markov_cli.read_documents(resolve_reference([str(corpus_dir)])), "plain", 1
        )
        assert any("sparsity" in w for w in report["warnings"])

    def test_large_corpus_is_not_flagged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(markov_cli, "MIN_REFERENCE_TOKENS", 10)
        source = tmp_path / "big.txt"
        source.write_text("the cat sat on the mat and the dog sat too\n")
        report = corpus_report(markov_cli.read_documents([str(source)]), "plain", 1)
        assert report["warnings"] == []

    def test_corpus_shorter_than_the_order_fails_cleanly(self, tmp_path):
        source = tmp_path / "tiny.txt"
        source.write_text("a b c\n")
        with pytest.raises(SystemExit) as excinfo:
            corpus_report(markov_cli.read_documents([str(source)]), "plain", 5)
        assert "too short" in str(excinfo.value)


class TestTextReport:
    def test_reports_composition_and_a_row_per_order(self, corpus_dir, capsys):
        assert main(["stats", "--reference", str(corpus_dir), "--order", "2"]) == 0
        out = capsys.readouterr().out

        assert "3 documents, 17 tokens, 9 distinct (plain tokenizer)" in out
        assert len(table_rows(out)) == 2

    def test_forced_share_is_shown_as_a_percentage(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir), "--order", "1"])
        assert "%" in capsys.readouterr().out

    def test_warning_reaches_the_terminal(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir), "--order", "1"])
        assert "warning:" in capsys.readouterr().out

    def test_single_document_is_not_pluralized(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir / "a.txt"), "--order", "1"])
        assert "1 document," in capsys.readouterr().out

    def test_defaults_to_order_3_and_the_plain_tokenizer(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir)])
        out = capsys.readouterr().out
        assert "plain tokenizer" in out
        assert len(table_rows(out)) == 3


class TestJsonReport:
    def test_emits_parseable_json(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir), "--report", "json"])
        report = json.loads(capsys.readouterr().out)
        assert report["tokens"] == 17
        assert report["tokenizer"] == "plain"
        assert len(report["orders"]) == 3

    def test_json_carries_per_document_paths(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir), "--report", "json"])
        report = json.loads(capsys.readouterr().out)
        assert [d["path"].rsplit("/", 1)[-1] for d in report["documents"]] == [
            "a.txt",
            "b.txt",
            "c.txt",
        ]

    def test_output_is_identical_across_runs(self, corpus_dir, capsys):
        main(["stats", "--reference", str(corpus_dir), "--report", "json"])
        first = capsys.readouterr().out
        main(["stats", "--reference", str(corpus_dir), "--report", "json"])
        assert capsys.readouterr().out == first


class TestOptionValidation:
    @pytest.mark.parametrize("order", ["0", "-1"])
    def test_order_below_one_exits_2(self, corpus_dir, order):
        with pytest.raises(SystemExit) as excinfo:
            main(["stats", "--reference", str(corpus_dir), "--order", order])
        assert excinfo.value.code == 2

    def test_non_integer_order_exits_2(self, corpus_dir):
        with pytest.raises(SystemExit) as excinfo:
            main(["stats", "--reference", str(corpus_dir), "--order", "two"])
        assert excinfo.value.code == 2

    def test_unknown_tokenizer_exits_2(self, corpus_dir):
        with pytest.raises(SystemExit) as excinfo:
            main(["stats", "--reference", str(corpus_dir), "--tokenizer", "legal"])
        assert excinfo.value.code == 2

    def test_unknown_report_format_exits_2(self, corpus_dir):
        with pytest.raises(SystemExit) as excinfo:
            main(["stats", "--reference", str(corpus_dir), "--report", "xml"])
        assert excinfo.value.code == 2


class TestGenerateTextSurvives:
    def test_generation_is_still_available_to_markov_py(self, corpus):
        """generate_text stays the engine behind markov.py until the generate
        subcommand lands at M6; test_generate_text.py pins its behavior."""
        produced = markov_cli.generate_text(corpus, 10).split()
        assert set(produced) <= set(" ".join(CORPUS_LINES).split())

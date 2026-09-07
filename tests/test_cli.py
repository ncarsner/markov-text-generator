"""Tests for the command-line layer: argument handling, output, file export."""

import random
import sys
import textwrap

import pytest

import markov_cli
from markov_cli import main

from conftest import CORPUS_WORDS

USAGE = "Usage: python markov_cli.py <input_file> <num_words or 0> [output_file]"


class TestArgumentValidation:
    @pytest.mark.parametrize("argv", [[], ["only_one_arg"]], ids=["no args", "one arg"])
    def test_too_few_arguments_exits_1_with_usage(self, argv, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 1
        assert capsys.readouterr().out.strip() == USAGE

    def test_non_integer_word_count_raises(self, corpus_file):
        with pytest.raises(ValueError):
            main([str(corpus_file), "not-a-number"])

    def test_missing_input_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main([str(tmp_path / "nope.txt"), "10"])

    def test_extra_arguments_are_ignored(self, corpus_file, tmp_path, capsys):
        out_file = tmp_path / "out.txt"
        main([str(corpus_file), "10", str(out_file), "ignored", "also-ignored"])
        assert out_file.exists()
        assert capsys.readouterr().out.strip()

    def test_falls_back_to_sys_argv(self, corpus_file, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["markov_cli.py", str(corpus_file), "10"])
        main()
        assert capsys.readouterr().out.strip()


class TestNumWords:
    def test_zero_means_input_length(self, corpus_file, capsys):
        """0 is the CLI's sentinel for 'as many words as the source has'."""
        main([str(corpus_file), "0"])
        produced = capsys.readouterr().out.split()
        assert len(produced) > 2
        assert len(produced) <= len(CORPUS_WORDS) + 2

    def test_explicit_count_bounds_the_output(self, corpus_file, capsys):
        main([str(corpus_file), "5"])
        assert len(capsys.readouterr().out.split()) <= 7


class TestStdout:
    def test_prints_words_drawn_from_the_source(self, corpus_file, capsys):
        main([str(corpus_file), "20"])
        assert set(capsys.readouterr().out.split()) <= set(CORPUS_WORDS)

    def test_output_is_wrapped_to_70_columns(self, tmp_path, capsys):
        source = tmp_path / "long.txt"
        source.write_text("The quick brown fox jumps over the lazy dog again.\n" * 40)
        main([str(source), "300"])
        printed = capsys.readouterr().out.rstrip("\n")
        assert "\n" in printed, "expected the text to wrap onto multiple lines"
        assert all(len(line) <= 70 for line in printed.split("\n"))


class TestFileExport:
    def test_no_file_written_without_a_third_argument(self, corpus_file, tmp_path):
        main([str(corpus_file), "10"])
        assert list(tmp_path.iterdir()) == [corpus_file]

    def test_writes_the_output_file(self, corpus_file, tmp_path):
        out_file = tmp_path / "out.txt"
        main([str(corpus_file), "10", str(out_file)])
        assert set(out_file.read_text().split()) <= set(CORPUS_WORDS)

    def test_overwrites_an_existing_file(self, corpus_file, tmp_path):
        out_file = tmp_path / "out.txt"
        out_file.write_text("stale content that must not survive")
        main([str(corpus_file), "10", str(out_file)])
        assert "stale" not in out_file.read_text()

    def test_file_holds_the_unwrapped_text_stdout_holds_the_wrapped_text(
        self, tmp_path, capsys
    ):
        """The file gets one long line; only stdout is passed through fill()."""
        source = tmp_path / "long.txt"
        source.write_text("The quick brown fox jumps over the lazy dog again.\n" * 40)
        out_file = tmp_path / "out.txt"

        main([str(source), "300", str(out_file)])

        printed = capsys.readouterr().out.rstrip("\n")
        written = out_file.read_text()
        assert "\n" not in written
        assert "\n" in printed
        assert written.split() == printed.split()

    def test_file_and_stdout_come_from_the_same_generation(self, corpus_file, tmp_path, capsys):
        out_file = tmp_path / "out.txt"
        main([str(corpus_file), "10", str(out_file)])
        assert out_file.read_text().split() == capsys.readouterr().out.split()


class TestEntryPoint:
    def test_main_is_wired_to_generate_text(self, corpus_file, capsys, monkeypatch):
        calls = []

        def spy(input_text, num_words=None):
            calls.append((input_text, num_words))
            return "Sentinel output"

        monkeypatch.setattr(markov_cli, "generate_text", spy)
        main([str(corpus_file), "0"])

        (input_text, num_words) = calls[0]
        assert num_words is None, "0 should reach generate_text as None"
        assert "".join(input_text) == corpus_file.read_text()
        assert capsys.readouterr().out.strip() == "Sentinel output"

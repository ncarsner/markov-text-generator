"""Tests for the corpus rebuild script.

The parsing functions are pure, so everything here runs without network; the
end-to-end cases feed main() a file:// URL.
"""

import pytest

from scripts.fetch_corpus import (
    PREFIX,
    build_corpus,
    clean_address,
    main,
    slugify,
    split_addresses,
    strip_boilerplate,
)

FILLER = " ".join(["liberty"] * 250)

EBOOK = f"""The Project Gutenberg eBook of Something
This boilerplate is separately licensed and must not reach the corpus.

*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***

INAUGURAL ADDRESSES OF THE PRESIDENTS

* * * * *

GEORGE WASHINGTON, FIRST INAUGURAL ADDRESS

IN THE CITY OF NEW YORK THURSDAY, APRIL 30, 1789

[Transcriber's note: this note spans
several lines and is a modern addition.]

Fellow-Citizens: {FILLER}

* * * * *

JOHN ADAMS, INAUGURAL ADDRESS

ON SATURDAY, MARCH 4, 1797

When it was first perceived: {FILLER}

*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***

This trailing boilerplate must not reach the corpus either.
"""


class TestStripBoilerplate:
    def test_keeps_only_the_marked_body(self):
        body = strip_boilerplate(EBOOK)
        assert "separately licensed" not in body
        assert "trailing boilerplate" not in body
        assert "Fellow-Citizens" in body

    @pytest.mark.parametrize(
        "text", ["no markers at all", "*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody"]
    )
    def test_missing_markers_raise(self, text):
        with pytest.raises(ValueError, match="markers not found"):
            strip_boilerplate(text)


class TestSplitAddresses:
    def test_splits_on_the_asterisk_rule(self):
        chunks = split_addresses(strip_boilerplate(EBOOK))
        assert len(chunks) == 3  # heading + two addresses


class TestSlugify:
    @pytest.mark.parametrize(
        "title, expected",
        [
            ("GEORGE WASHINGTON, FIRST INAUGURAL ADDRESS",
             "george_washington_first_inaugural_address"),
            ("WILLIAM J. CLINTON", "william_j_clinton"),
            ("  Spaced  Out  ", "spaced_out"),
        ],
    )
    def test_normalizes_titles(self, title, expected):
        assert slugify(title) == expected


class TestCleanAddress:
    def address(self, chunk=None):
        return clean_address(chunk or split_addresses(strip_boilerplate(EBOOK))[1])

    def test_returns_filename_and_text(self):
        name, text = self.address()
        assert name == f"{PREFIX}george_washington_first_inaugural_address.txt"
        assert text.startswith("Fellow-Citizens:")

    def test_drops_the_transcriber_note(self):
        _, text = self.address()
        assert "Transcriber" not in text
        assert "modern addition" not in text

    def test_drops_the_dateline_and_title(self):
        _, text = self.address()
        assert "CITY OF NEW YORK" not in text
        assert "GEORGE WASHINGTON" not in text

    def test_collapses_to_a_single_line(self):
        _, text = self.address()
        assert text.count("\n") == 1
        assert text.endswith("\n")

    def test_strips_the_gutenberg_attribution(self):
        chunk = (
            "WILLIAM JEFFERSON CLINTON, FIRST INAUGURAL ADDRESS\n\n"
            "[As presented on the Internet by Project Gutenberg on January 20th, 1993]\n\n"
            f"My fellow citizens: {FILLER}\n"
        )
        _, text = clean_address(chunk)
        assert "Project Gutenberg" not in text
        assert text.startswith("My fellow citizens:")

    @pytest.mark.parametrize(
        "chunk, reason",
        [
            ("", "empty chunk"),
            ("INAUGURAL ADDRESSES OF THE PRESIDENTS\n\nfront matter", "not an address"),
            ("JOHN DOE, INAUGURAL ADDRESS\n\ntoo short to keep", "under the word floor"),
        ],
    )
    def test_rejects_non_addresses(self, chunk, reason):
        assert clean_address(chunk) is None


class TestBuildCorpus:
    def test_returns_every_address(self):
        built = build_corpus(EBOOK)
        assert [name for name, _ in built] == [
            f"{PREFIX}george_washington_first_inaugural_address.txt",
            f"{PREFIX}john_adams_inaugural_address.txt",
        ]


class TestMain:
    @pytest.fixture
    def ebook_url(self, tmp_path):
        source = tmp_path / "ebook.txt"
        source.write_text(EBOOK, encoding="utf-8")
        return source.as_uri()

    def test_writes_the_corpus(self, ebook_url, tmp_path, capsys):
        dest = tmp_path / "corpus"
        assert main(["--url", ebook_url, "--dest", str(dest)]) == 0
        written = sorted(p.name for p in dest.glob("*.txt"))
        assert len(written) == 2
        assert "Wrote 2 addresses" in capsys.readouterr().out

    def test_refuses_to_clobber_without_force(self, ebook_url, tmp_path, capsys):
        dest = tmp_path / "corpus"
        dest.mkdir()
        (dest / f"{PREFIX}existing.txt").write_text("keep me", encoding="utf-8")

        assert main(["--url", ebook_url, "--dest", str(dest)]) == 0

        assert (dest / f"{PREFIX}existing.txt").read_text() == "keep me"
        assert "pass --force" in capsys.readouterr().out

    def test_force_replaces_stale_files(self, ebook_url, tmp_path):
        dest = tmp_path / "corpus"
        dest.mkdir()
        stale = dest / f"{PREFIX}stale.txt"
        stale.write_text("obsolete", encoding="utf-8")

        assert main(["--url", ebook_url, "--dest", str(dest), "--force"]) == 0

        assert not stale.exists()
        assert len(list(dest.glob("*.txt"))) == 2

    def test_leaves_unrelated_corpus_files_alone(self, ebook_url, tmp_path):
        """The two pre-existing speeches do not carry the prefix and must survive."""
        dest = tmp_path / "corpus"
        dest.mkdir()
        other = dest / "speech_day_of_infamy.txt"
        other.write_text("Yesterday, December 7th, 1941", encoding="utf-8")

        main(["--url", ebook_url, "--dest", str(dest), "--force"])

        assert other.read_text() == "Yesterday, December 7th, 1941"

    def test_reports_failure_when_nothing_parses(self, tmp_path, capsys):
        source = tmp_path / "empty.txt"
        source.write_text(
            "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\nnothing here\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n",
            encoding="utf-8",
        )
        assert main(["--url", source.as_uri(), "--dest", str(tmp_path / "out")]) == 1
        assert "format may have changed" in capsys.readouterr().err

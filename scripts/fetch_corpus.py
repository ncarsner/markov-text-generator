#!/usr/bin/env python3
"""Rebuild the public-domain evaluation corpus in data/input/.

The corpus is deliberately untracked (see the blanket ``*.txt`` rule in
.gitignore), so this script regenerates it from its upstream source.

It downloads Project Gutenberg ebook #925 -- the presidential inaugural
addresses, 1789-2005 -- and writes one file per address. Every address is a
work of the United States federal government and is uncopyrighted under
17 U.S.C. 105. Gutenberg's own boilerplate is separately licensed and is
stripped, along with transcriber's notes and place/date headers, so what
lands on disk is speech text only.

The two speeches that predate this script (speech_day_of_infamy.txt and
speech_we_choose_to_go_to_the_moon.txt) are left alone.

Usage:
    uv run python scripts/fetch_corpus.py
    uv run python scripts/fetch_corpus.py --dest data/input --force
"""

import argparse
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

EBOOK_URL = "https://www.gutenberg.org/cache/epub/925/pg925.txt"
PREFIX = "speech_inaugural_"

# An address shorter than this is a fragment or a table-of-contents entry.
MIN_WORDS = 200

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

ADDRESS_DELIMITER = re.compile(r"\n\s*\*\s\*\s\*\s\*\s\*\s*\n")
TRANSCRIBER_NOTE = re.compile(r"\[Transcriber.*?\]", re.S)
PG_ATTRIBUTION = re.compile(r"\[As presented on the Internet by Project Gutenberg[^\]]*\]\s*")
DATELINE = re.compile(
    r"^\s*(IN|ON|AT|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\b.*$",
    re.M | re.I,
)


def strip_boilerplate(raw):
    """Return only the text between Gutenberg's START and END markers.

    The surrounding boilerplate carries the Project Gutenberg License; the
    speeches themselves are federal works in the public domain.
    """
    if START_MARKER not in raw or END_MARKER not in raw:
        raise ValueError("Gutenberg markers not found -- is this the expected ebook?")
    body = raw.split(START_MARKER, 1)[1].split("\n", 1)[1]
    return body.split(END_MARKER, 1)[0]


def split_addresses(body):
    """Split the ebook body into per-address chunks on its '* * * * *' rules."""
    return ADDRESS_DELIMITER.split(body)


def slugify(title):
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_title.lower()).strip("_")


def clean_address(chunk):
    """Reduce one chunk to (filename, speech text), or None if it is not an address.

    Drops the title and dateline, transcriber's notes, and the stray Gutenberg
    attribution in the 1993 Clinton address, then collapses the remainder to a
    single line.
    """
    lines = [line.rstrip() for line in chunk.strip().split("\n")]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return None

    title = lines[0].strip()
    if "INAUGURAL ADDRESS" not in title.upper():
        return None

    rest = "\n".join(lines[1:])
    rest = TRANSCRIBER_NOTE.sub("", rest)
    rest = DATELINE.sub("", rest)
    if len(rest.split()) < MIN_WORDS:
        return None

    text = " ".join(rest.split("\n")).strip()
    text = PG_ATTRIBUTION.sub("", text)
    return f"{PREFIX}{slugify(title)}.txt", text + "\n"


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "markov-text-generator/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def build_corpus(raw):
    """Turn the raw ebook into a list of (filename, text) pairs."""
    addresses = []
    for chunk in split_addresses(strip_boilerplate(raw)):
        cleaned = clean_address(chunk)
        if cleaned is not None:
            addresses.append(cleaned)
    return addresses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=Path("data/input"),
                        help="directory to write the corpus into (default: data/input)")
    parser.add_argument("--url", default=EBOOK_URL, help="source ebook URL")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files instead of stopping")
    args = parser.parse_args(argv)

    existing = sorted(args.dest.glob(f"{PREFIX}*.txt")) if args.dest.is_dir() else []
    if existing and not args.force:
        print(f"{len(existing)} address files already in {args.dest}; pass --force to rebuild.")
        return 0

    print(f"Downloading {args.url} ...", file=sys.stderr)
    addresses = build_corpus(download(args.url))
    if not addresses:
        print("No addresses parsed -- the upstream format may have changed.", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    for stale in existing:
        stale.unlink()

    total = 0
    for name, text in addresses:
        (args.dest / name).write_text(text, encoding="utf-8")
        total += len(text.split())

    print(f"Wrote {len(addresses)} addresses ({total:,} words) to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

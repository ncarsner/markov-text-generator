"""Command line: subcommand dispatch over the markov_core engine.

`stats` is the first subcommand to land. It reports whether a reference corpus
is large and varied enough to be worth scoring against, which is the question
that gates every other use of the tool (see docs/PRD-deviation-analysis.md).

`analyze` and `generate` follow at later milestones. Until `generate` lands,
markov.py remains the front door for text generation; generate_text() below is
the engine behind it and keeps its original behavior exactly.
"""

import argparse
import glob
import json
import os
import random
import sys

import markov_core

ORDER = 2

# The README states a reference corpus of roughly 10^5 tokens as the point
# below which scores are dominated by sparsity rather than by the corpus.
# Measured: at 2,648 tokens the chain is 92% forced at order 2 and can only
# transcribe itself.
MIN_REFERENCE_TOKENS = 100_000


def generate_text(input_text, num_words=None):
    """Generate text from an order-2 chain over ``input_text`` (a list of lines).

    A thin wrapper over markov_core that preserves this function's original
    behavior exactly, quirks included: the walk is seeded from a capitalized
    context, and the two sentinel entries below make it loop back to the start
    of the corpus rather than stop at the end. Both are pinned by tests.
    """
    words = [word for line in input_text for word in line.split()]

    # Padding with two empty tokens reproduces the original's starting state,
    # in which the first words of the corpus follow an empty context.
    padded = ["", ""] + words
    chain = markov_core.build_chain(padded, ORDER)

    # Close the chain at the end of the corpus. These are what make the walk
    # wrap around to the first word instead of terminating.
    chain[padded[-2], padded[-1]].append("")
    chain[padded[-1], ""].append("")

    if num_words is None:
        num_words = len(words)

    start = random.choice(markov_core.capitalized_contexts(chain))
    return " ".join(markov_core.sample(chain, start, num_words, ORDER))


# --- Reference corpus ------------------------------------------------------

def resolve_reference(patterns):
    """Expand reference patterns into a sorted list of file paths.

    Each pattern may be a directory (every *.txt inside it), a glob, or a
    single file. Globs are expanded here as well as by the shell so that a
    quoted pattern -- the form the README recommends, since a bare *.txt sweeps
    up whatever else a working data/input/ happens to hold -- behaves the same.
    """
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            matches = sorted(glob.glob(os.path.join(pattern, "*.txt")))
            if not matches:
                raise SystemExit(f"no .txt files in directory: {pattern}")
        elif any(char in pattern for char in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise SystemExit(f"pattern matched no files: {pattern}")
        else:
            if not os.path.isfile(pattern):
                raise SystemExit(f"no such file: {pattern}")
            matches = [pattern]
        paths.extend(matches)

    # Deduplicate while keeping order: overlapping patterns must not count a
    # document twice and silently distort the totals.
    seen = set()
    unique = [p for p in paths if not (p in seen or seen.add(p))]
    if not unique:
        raise SystemExit("empty reference corpus")
    return unique


def read_documents(paths):
    """Read each path as UTF-8, returning a list of (path, text) pairs.

    Encoding is explicit: the locale default fails on a corpus containing smart
    quotes or em-dashes, which most real prose does.
    """
    documents = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            documents.append((path, handle.read()))
    return documents


# --- stats -----------------------------------------------------------------

def corpus_report(documents, tokenizer_name, max_order):
    """Measure a corpus: composition, plus branching and forced share by order.

    Documents are concatenated before counting, which is how the figures in
    README.md and docs/ROADMAP.md were measured and how the analysis path
    builds its model. It admits a small number of n-grams that straddle a
    document boundary; at corpus scale the effect is below the reported
    precision.
    """
    tokenizer = markov_core.get_tokenizer(tokenizer_name)
    per_document = [
        {"path": path, "tokens": len(tokenizer(text))} for path, text in documents
    ]
    tokens = tokenizer(" ".join(text for _, text in documents))

    orders = []
    for order in range(1, max_order + 1):
        # A corpus shorter than the order has no states at all; say so rather
        # than surfacing the engine's exception.
        try:
            branching, forced = markov_core.chain_statistics(tokens, order)
        except ValueError:
            raise SystemExit(
                f"corpus of {len(tokens):,} tokens is too short to report "
                f"order {order}"
            ) from None
        orders.append(
            {
                "order": order,
                "branching": round(branching, 6),
                "forced": round(forced, 6),
            }
        )

    warnings = []
    if len(tokens) < MIN_REFERENCE_TOKENS:
        warnings.append(
            f"reference corpus is {len(tokens):,} tokens; below roughly "
            f"{MIN_REFERENCE_TOKENS:,} scores are dominated by sparsity"
        )

    return {
        "tokenizer": tokenizer_name,
        "documents": per_document,
        "tokens": len(tokens),
        "vocabulary": len(set(tokens)),
        "orders": orders,
        "warnings": warnings,
    }


def format_report(report, patterns):
    """Render a corpus report as the human-readable table."""
    count = len(report["documents"])
    lines = [
        f"reference: {' '.join(patterns)}",
        f"           {count} document{'s' if count != 1 else ''}, "
        f"{report['tokens']:,} tokens, {report['vocabulary']:,} distinct "
        f"({report['tokenizer']} tokenizer)",
        "",
        "  order   branching   forced",
    ]
    for row in report["orders"]:
        lines.append(
            f"  {row['order']:>5}   {row['branching']:>9.2f}   {row['forced']:>5.0%}"
        )
    lines.append("")
    lines.append(
        "  branching: distinct words that can follow a context, averaged\n"
        "  forced:    share of contexts with only one continuation"
    )
    for warning in report["warnings"]:
        lines.append("")
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


def cmd_stats(args):
    paths = resolve_reference(args.reference)
    report = corpus_report(read_documents(paths), args.tokenizer, args.order)

    if args.report == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report, args.reference))

    return 0


# --- Dispatch --------------------------------------------------------------

def positive_int(value):
    """An argparse type for arguments that must be 1 or greater."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {number}")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        prog="markov",
        description="Measure how far a document's language deviates from a "
        "reference corpus.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    stats = subcommands.add_parser(
        "stats",
        help="report whether a reference corpus is large enough to trust",
        description="Report corpus composition, branching factor, and the "
        "share of forced states by chain order.",
    )
    stats.add_argument(
        "--reference",
        nargs="+",
        required=True,
        metavar="PATH",
        help="reference corpus: directories, globs, or files",
    )
    stats.add_argument(
        "--order",
        type=positive_int,
        default=markov_core.DEFAULT_ORDER,
        help="report orders 1 through N (default: %(default)s)",
    )
    stats.add_argument(
        "--tokenizer",
        choices=sorted(markov_core.TOKENIZERS),
        default="plain",
        help="plain lowercases and drops punctuation for analysis; "
        "whitespace preserves both, the generator's view (default: %(default)s)",
    )
    stats.add_argument(
        "--report",
        choices=("text", "json"),
        default="text",
        help="output format (default: %(default)s)",
    )
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

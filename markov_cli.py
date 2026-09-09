"""Command line: subcommand dispatch over the markov_core engine.

`stats` reports whether a reference corpus is large and varied enough to be
worth scoring against, which is the question that gates everything else.
`analyze` is the principal application: it scores how far a document's language
deviates from that corpus (see docs/PRD-deviation-analysis.md).

`generate` follows at a later milestone. Until it lands, markov.py remains the
front door for text generation; generate_text() below is the engine behind it
and keeps its original behavior exactly.
"""

import argparse
import glob
import json
import os
import random
import statistics
import sys

import markov_core

ORDER = 2

# The README states a reference corpus of roughly 10^5 tokens as the point
# below which scores are dominated by sparsity rather than by the corpus.
# Measured: at 2,648 tokens the chain is 92% forced at order 2 and can only
# transcribe itself.
MIN_REFERENCE_TOKENS = 100_000

# Beyond this many standard deviations a document is called variant. The README
# reads |z| < 1 as typical, 1-2 as ordinary variation, and beyond 2.5 as worth a
# look; this is that last boundary.
VARIANT_Z = 2.5

# --baseline holdout scores every fifth document against the rest. Spread
# through the corpus rather than taken as a contiguous block: a reference
# ordered by date would otherwise hold out one era and call it the expectation.
HOLDOUT_EVERY = 5


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


# --- analyze ---------------------------------------------------------------

def measure(model, tokens):
    """Score one document under one model: surprisal and novel n-gram rate."""
    return {
        "bits_per_token": model.bits_per_token(tokens),
        "novel_ngram_rate": model.novel_ngram_rate(tokens),
    }


def calibrate(token_lists, order, baseline):
    """Measure what surprisal this reference corpus expects of its own members.

    This is what makes deviation calculable rather than merely observed: a
    document's score means nothing until there is a distribution to read it
    against, and the only honest source for that distribution is the reference
    itself. Leave-one-out scores every document against a model built from all
    the others; holdout builds one model and is cheaper on a large corpus.

    Spread is the sample standard deviation, since the reference is a sample of
    the language being characterized rather than the whole of it.
    """
    if len(token_lists) < 2:
        raise SystemExit(
            "calibration needs at least 2 reference documents; "
            f"got {len(token_lists)}"
        )

    scores = []
    trained_on = []
    if baseline == "loo":
        for held, tokens in enumerate(token_lists):
            others = [t for i, t in enumerate(token_lists) if i != held]
            model = markov_core.NgramModel(
                [token for document in others for token in document], order=order
            )
            scores.append(measure(model, tokens))
    else:
        held_out = list(range(0, len(token_lists), HOLDOUT_EVERY))
        trained_on = [t for i, t in enumerate(token_lists) if i not in set(held_out)]
        if not trained_on or len(held_out) < 2:
            raise SystemExit(
                f"holdout needs at least {HOLDOUT_EVERY * 2} reference documents; "
                f"got {len(token_lists)} -- use --baseline loo"
            )
        model = markov_core.NgramModel(
            [token for document in trained_on for token in document], order=order
        )
        scores.append(measure(model, token_lists[held_out[0]]))
        scores.extend(measure(model, token_lists[i]) for i in held_out[1:])

    def distribution(key):
        values = [score[key] for score in scores]
        return {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values),
        }

    return {
        "method": "leave-one-out" if baseline == "loo" else "holdout",
        "documents": len(scores),
        "trained_on": len(token_lists) - 1 if baseline == "loo" else len(trained_on),
        "surprisal": distribution("bits_per_token"),
        "novel_ngram_rate": distribution("novel_ngram_rate"),
        # Kept so a document can be ranked against the reference directly.
        # A count of documents it out-scores needs no statistics to read, and
        # unlike z it cannot be distorted by a lopsided distribution.
        "scores": sorted(score["bits_per_token"] for score in scores),
    }


def deviation(value, expected):
    """Standard deviations between a score and its expected distribution."""
    if expected["stdev"] == 0:
        raise SystemExit(
            "reference documents all score identically; no spread to measure "
            "deviation against"
        )
    return (value - expected["mean"]) / expected["stdev"]


def analyze_documents(reference, targets, tokenizer_name, order, baseline):
    """Score each target against the reference, calibrated against the reference.

    A target that is itself part of the reference is scored against a model
    built without it. Left in, it would be scoring itself and the corpus would
    flatter it.
    """
    tokenizer = markov_core.get_tokenizer(tokenizer_name)
    reference_tokens = [tokenizer(text) for _, text in reference]
    reference_paths = [path for path, _ in reference]

    calibration = calibrate(reference_tokens, order, baseline)
    whole = markov_core.NgramModel(
        [token for document in reference_tokens for token in document], order=order
    )

    scored = []
    for path, text in targets:
        tokens = tokenizer(text)
        if len(tokens) <= order:
            raise SystemExit(
                f"{path}: {len(tokens)} tokens is too short to score at order {order}"
            )

        held_out = path in reference_paths
        if held_out:
            keep = reference_paths.index(path)
            model = markov_core.NgramModel(
                [
                    token
                    for i, document in enumerate(reference_tokens)
                    if i != keep
                    for token in document
                ],
                order=order,
            )
        else:
            model = whole

        result = measure(model, tokens)
        scored.append(
            {
                "path": path,
                "tokens": len(tokens),
                "held_out": held_out,
                "bits_per_token": round(result["bits_per_token"], 6),
                "novel_ngram_rate": round(result["novel_ngram_rate"], 6),
                "z": round(
                    deviation(result["bits_per_token"], calibration["surprisal"]), 6
                ),
                "novel_ngram_z": round(
                    deviation(
                        result["novel_ngram_rate"], calibration["novel_ngram_rate"]
                    ),
                    6,
                ),
                "reference_documents_below": sum(
                    score < result["bits_per_token"] for score in calibration["scores"]
                ),
                "verdict": (
                    "variant"
                    if abs(
                        deviation(result["bits_per_token"], calibration["surprisal"])
                    )
                    > VARIANT_Z
                    else "typical"
                ),
            }
        )

    corpus_tokens = sum(len(document) for document in reference_tokens)
    warnings = []
    if corpus_tokens < MIN_REFERENCE_TOKENS:
        warnings.append(
            f"reference corpus is {corpus_tokens:,} tokens; below roughly "
            f"{MIN_REFERENCE_TOKENS:,} scores are dominated by sparsity"
        )

    return {
        "tokenizer": tokenizer_name,
        "order": order,
        "reference": {
            "documents": len(reference),
            "tokens": corpus_tokens,
            "paths": reference_paths,
        },
        "calibration": {
            "method": calibration["method"],
            "documents": calibration["documents"],
            "trained_on": calibration["trained_on"],
            "surprisal_mean": round(calibration["surprisal"]["mean"], 6),
            "surprisal_stdev": round(calibration["surprisal"]["stdev"], 6),
            "novel_ngram_mean": round(calibration["novel_ngram_rate"]["mean"], 6),
            "novel_ngram_stdev": round(calibration["novel_ngram_rate"]["stdev"], 6),
        },
        "documents": scored,
        "warnings": warnings,
    }


def _how_expected_was_measured(calibration):
    """Say where the expected range came from, without naming the technique.

    "Leave-one-out" is jargon that hides a simple idea: a corpus is asked what
    it expects of its own members. Reports are read by people deciding whether
    to act on a flag, so the output states the procedure and leaves the term to
    the JSON and the docs.
    """
    if calibration["method"] == "leave-one-out":
        return (
            f"measured by scoring each of the {calibration['documents']} reference "
            f"documents against the other {calibration['trained_on']}"
        )
    return (
        f"measured by scoring {calibration['documents']} reference documents "
        f"against the other {calibration['trained_on']}"
    )


def _row(label, number, suffix, note=""):
    """One aligned metric line: label, right-aligned number, unit, then note."""
    body = f"  {label:<16}{number:>7}  {suffix}"
    return f"{body:<40}{note}".rstrip()


def format_analysis(analysis, patterns):
    """Render an analysis as the human-readable report.

    The reference and its expected range are stated once, above the documents
    scored against them: they are the same for every document in one run, and a
    score is meaningless without them.
    """
    reference = analysis["reference"]
    calibration = analysis["calibration"]
    order = analysis["order"]

    lines = [
        f"reference:  {' '.join(patterns)}",
        f"            {reference['documents']} documents, "
        f"{reference['tokens']:,} tokens, "
        f"{analysis['tokenizer']} tokenizer, order {order}",
        f"expected:   {calibration['surprisal_mean']:.2f} +/- "
        f"{calibration['surprisal_stdev']:.2f} bits/token, "
        f"{calibration['novel_ngram_mean']:.0%} +/- "
        f"{calibration['novel_ngram_stdev']:.0%} novel {order}-grams",
        f"            {_how_expected_was_measured(calibration)}",
    ]

    for document in analysis["documents"]:
        lines.append("")
        lines.append(
            f"document:   {document['path']}  ({document['tokens']:,} tokens)"
        )
        lines.append(
            _row("surprisal", f"{document['bits_per_token']:.2f}", "bits/token")
        )
        lines.append(
            _row(
                "deviation",
                f"{document['z']:+.2f}",
                "z",
                document["verdict"].upper(),
            )
        )
        lines.append(
            _row(
                f"novel {order}-grams",
                f"{document['novel_ngram_rate']:.0%}",
                "",
                f"{document['novel_ngram_z']:+.2f} z",
            )
        )
        lines.append(
            f"  reads as more unusual than {document['reference_documents_below']} "
            f"of the {calibration['documents']} reference documents"
        )
        if document["held_out"]:
            lines.append(
                "  this document is part of the reference; scored against a "
                "model built without it"
            )

    lines.append("")
    # The PRD requires this in the output itself, not only in the docs: a
    # statistical outlier is not a finding, and the number invites being read
    # as one.
    lines.append(
        "  variant means unlike this reference corpus. It does not mean wrong,\n"
        "  poorly written, risky, or non-compliant, and \"expected\" only ever\n"
        "  means typical of the corpus you supplied."
    )
    for warning in analysis["warnings"]:
        lines.append("")
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


def cmd_analyze(args):
    reference = read_documents(resolve_reference(args.reference))
    targets = read_documents(resolve_reference(args.target))
    analysis = analyze_documents(
        reference, targets, args.tokenizer, args.order, args.baseline
    )

    if args.report == "json":
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print(format_analysis(analysis, args.reference))

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

    analyze = subcommands.add_parser(
        "analyze",
        help="score how far a document deviates from a reference corpus",
        description="Score one or more target documents against a reference "
        "corpus, calibrated against the corpus's expectation of its own "
        "members.",
    )
    analyze.add_argument(
        "target",
        nargs="+",
        metavar="TARGET",
        help="documents to score: files or globs",
    )
    analyze.add_argument(
        "--reference",
        nargs="+",
        required=True,
        metavar="PATH",
        help="reference corpus: directories, globs, or files",
    )
    analyze.add_argument(
        "--order",
        type=positive_int,
        default=markov_core.DEFAULT_ORDER,
        help="n-gram order (default: %(default)s)",
    )
    analyze.add_argument(
        "--baseline",
        choices=("loo", "holdout"),
        default="loo",
        help="how to calibrate the expected range: leave-one-out builds a model "
        "per reference document, holdout builds one (default: %(default)s)",
    )
    analyze.add_argument(
        "--tokenizer",
        choices=sorted(markov_core.TOKENIZERS),
        default="plain",
        help="see stats --tokenizer (default: %(default)s)",
    )
    analyze.add_argument(
        "--report",
        choices=("text", "json"),
        default="text",
        help="output format (default: %(default)s)",
    )
    analyze.set_defaults(func=cmd_analyze)

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

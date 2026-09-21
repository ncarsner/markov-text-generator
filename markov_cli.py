"""Command line: subcommand dispatch over the markov_core engine.

`stats` reports whether a reference corpus is large and varied enough to be
worth scoring against, which is the question that gates everything else.
`analyze` is the principal application: it scores how far a document's language
deviates from that corpus (see docs/PRD-deviation-analysis.md).

`generate` walks the same counts forwards instead of scoring against them. It is
the by-product, not the point.
"""

import argparse
import contextlib
import glob
import json
import math
import os
import random
import re
import statistics
import sys
import textwrap

import markov_core

# The README states a reference corpus of roughly 10^5 tokens as the point
# below which scores are dominated by sparsity rather than by the corpus.
# Measured: at 2,648 tokens the chain is 92% forced at order 2 and can only
# transcribe itself.
MIN_REFERENCE_TOKENS = 100_000

# Beyond this many standard deviations a document is called variant, once the
# reference is large enough for that to be the right number. The README reads
# |z| < 1 as typical, 1-2 as ordinary variation, and beyond 2.5 as worth a
# look; this is that last boundary, and the limit the size-adjusted threshold
# converges to. See verdict_threshold for why it cannot be applied flat.
VARIANT_Z = 2.5

# A reference document has to clear its threshold without passing the ceiling
# that verdict_ceiling describes, so how close those two sit is how much room
# the test has to work in. Measured by simulation: where the threshold reaches
# 90% of the ceiling -- 6 documents -- a member sitting four standard
# deviations from its peers is caught 42% of the time. At 8 documents (82%)
# that is 61%, and at 54 (34%) it is 91%. Below this the report says so.
CROWDED_CEILING = 0.9

# --baseline holdout scores every fifth document against the rest. Spread
# through the corpus rather than taken as a contiguous block: a reference
# ordered by date would otherwise hold out one era and call it the expectation.
HOLDOUT_EVERY = 5


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


def calibrate(token_lists, order, baseline, model=None):
    """Measure what surprisal this reference corpus expects of its own members.

    This is what makes deviation calculable rather than merely observed: a
    document's score means nothing until there is a distribution to read it
    against, and the only honest source for that distribution is the reference
    itself. Leave-one-out scores every document against a model built from all
    the others; holdout builds one model and scores only a fifth of them.

    Leave-one-out subtracts each document's counts from one whole-corpus model
    rather than rebuilding, which is exact rather than approximate: see
    ``NgramModel.without``. ``model`` is that whole-corpus model, built here if
    the caller has not built one already.

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
        if model is None:
            model = markov_core.NgramModel.from_documents(token_lists, order=order)
        for held, tokens in enumerate(token_lists):
            with model.without(held) as others:
                scores.append(measure(others, tokens))
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


def verdict_ceiling(documents):
    """The largest |z| a member of the reference can reach, given how many.

    A reference document is standardized against a mean and spread computed
    from the same set of scores it belongs to, which bounds it at
    (n-1)/sqrt(n). The bound is arithmetic, not statistical: it holds however
    long the documents are and however far the language varies.

    It matters because it sits below VARIANT_Z until 9 documents: a flat 2.5
    was unreachable there, and "typical" said only that the corpus was too
    small to say otherwise. ``verdict_threshold`` derives the threshold from
    this bound instead, so what a member must clear is always inside what it
    can reach. How close the two sit is how much room the test has, which is
    what the ceiling is still reported for.

    An outside target is not bounded this way: it is standardized against
    scores it is not part of.
    """
    return (documents - 1) / math.sqrt(documents)


def _beta_continued_fraction(a, b, x):
    """Lentz's method for the continued fraction of the incomplete beta."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    fraction = d
    for m in range(1, 300):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            step = d * c
            fraction *= step
        if abs(step - 1.0) < 1e-15:
            break
    return fraction


def _regularized_beta(a, b, x):
    """I_x(a, b), the regularized incomplete beta function.

    Present because Student's t is what a spread estimated from a sample calls
    for, and the standard library stops at the normal distribution. Only the t
    distribution function needs it, and only twice per run.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    # The fraction converges quickly on one side of this point and slowly on
    # the other; the symmetry I_x(a,b) = 1 - I_(1-x)(b,a) takes the fast side.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _t_critical(alpha, df):
    """The t value a sample of df + 1 exceeds in absolute terms alpha of the time."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(|T| > mid), rising as mid falls, so the bisection walks up.
        if _regularized_beta(df / 2.0, 0.5, df / (df + mid * mid)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def variant_alpha():
    """The false-flag rate VARIANT_Z implies, if the expected range were known.

    1.24%, or about one document in 80. The size-adjusted threshold holds this
    rate fixed at every reference size, which is the whole of what it does.
    Derived here rather than stored beside VARIANT_Z so that the two cannot
    drift apart: VARIANT_Z stays the one number that sets how readily the tool
    calls something variant.
    """
    return 2 * (1 - statistics.NormalDist().cdf(VARIANT_Z))


def verdict_threshold(documents, held_out):
    """How far a document must deviate to be called variant, given how many.

    A flat 2.5 assumes the expected range is known. It is estimated, from as
    few as a handful of scores, and the error in that estimate runs the wrong
    way in both directions at once:

    A document inside the reference is squeezed by ``verdict_ceiling``. At 5
    documents it cannot pass 1.79, so 2.5 called nothing variant, ever. The
    threshold here is the matching quantile of the studentized residual, which
    is bounded by the ceiling in the same way and so always sits below it: 1.70
    at 5 documents, of a possible 1.79.

    A target from outside is squeezed the other way. Its z is divided by a
    spread estimated from few enough documents to be badly wrong, so 2.5 fired
    on noise -- measured on 5-document subsets, the same speech was called
    variant 18 times in 60. The threshold here is the prediction-interval form,
    t(n-1) * sqrt(1 + 1/n): 4.74 at 5 documents.

    Both hold the false-flag rate at variant_alpha() whatever the size, both
    converge to VARIANT_Z as the reference grows, so a large corpus reads as it
    always did. Verified by simulation against the nominal rate; see the tests.
    """
    if held_out:
        ceiling = verdict_ceiling(documents)
        if documents < 3:
            # Two scores sit at +/- 0.707 by construction, whatever they are.
            # Nothing to estimate a spread from, so nothing can be called
            # variant: the ceiling is unreachable, the verdict strict.
            #
            # Removing this guard is an equivalent mutation, but only by
            # accident: at zero degrees of freedom the bisection bottoms out at
            # t = 3e-57 rather than 0, so t^2/(0 + t^2) is exactly 1 and the
            # line below returns the same ceiling. Widen the search range and
            # that becomes 0/0. The guard says what is meant instead.
            return ceiling
        t = _t_critical(variant_alpha(), documents - 2)
        return ceiling * math.sqrt(t * t / (documents - 2 + t * t))
    return _t_critical(variant_alpha(), documents - 1) * math.sqrt(1 + 1 / documents)


def deviation(value, expected):
    """Standard deviations between a score and its expected distribution."""
    if expected["stdev"] == 0:
        raise SystemExit(
            "reference documents all score identically; no spread to measure "
            "deviation against"
        )
    return (value - expected["mean"]) / expected["stdev"]


def analyze_documents(
    reference, targets, tokenizer_name, order, baseline, window=12, top_spans=5
):
    """Score each target against the reference, calibrated against the reference.

    A target that is itself part of the reference is scored against a model
    built without it. Left in, it would be scoring itself and the corpus would
    flatter it.
    """
    tokenizer = markov_core.get_tokenizer(tokenizer_name)
    reference_tokens = [tokenizer(text) for _, text in reference]
    reference_paths = [path for path, _ in reference]

    # One model serves the whole run: calibration borrows it a document at a
    # time, and a target already in the reference is scored by the same
    # subtraction rather than by a rebuild of its own.
    whole = markov_core.NgramModel.from_documents(reference_tokens, order=order)
    calibration = calibrate(reference_tokens, order, baseline, model=whole)

    scored = []
    for path, text in targets:
        located = markov_core.locate_tokens(text, tokenizer_name)
        tokens = [token for token, _, _ in located]
        if len(tokens) <= order:
            raise SystemExit(
                f"{path}: {len(tokens)} tokens is too short to score at order {order}"
            )

        held_out = path in reference_paths
        threshold = verdict_threshold(calibration["documents"], held_out)
        borrowed = (
            whole.without(reference_paths.index(path))
            if held_out
            else contextlib.nullcontext(whole)
        )
        with borrowed as model:
            surprisals = model.surprisals(tokens)
            result = {
                "bits_per_token": sum(surprisals) / len(surprisals),
                "novel_ngram_rate": model.novel_ngram_rate(tokens),
            }
            # Spans are scored by the same model as the document, so a held-out
            # target's passages are not judged against a model that read them.
            spans = document_spans(
                model, text, located, surprisals, window, top_spans
            )
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
                "spans": spans,
                # What this document had to pass, not the flat VARIANT_Z: a
                # reference member and an outside target are held to different
                # numbers, and neither is the same at 5 documents as at 54.
                "threshold": round(threshold, 6),
                "verdict": (
                    "variant"
                    if abs(
                        deviation(result["bits_per_token"], calibration["surprisal"])
                    )
                    > threshold
                    else "typical"
                ),
            }
        )

    corpus_tokens = sum(len(document) for document in reference_tokens)
    ceiling = verdict_ceiling(calibration["documents"])
    warnings = []
    if corpus_tokens < MIN_REFERENCE_TOKENS:
        warnings.append(
            f"reference corpus is {corpus_tokens:,} tokens; below roughly "
            f"{MIN_REFERENCE_TOKENS:,} scores are dominated by sparsity"
        )
    # Independent of the token warning above: a corpus can be large in words
    # and still hold too few documents to say anything. Three long books clear
    # the token gate and leave a member needing 1.15 z of a possible 1.15.
    documents = calibration["documents"]
    member_threshold = verdict_threshold(documents, held_out=True)
    if member_threshold >= CROWDED_CEILING * ceiling:
        warnings.append(
            f"{documents} documents leave a reference document needing "
            f"{member_threshold:.2f} z of a possible {ceiling:.2f}, and an "
            f"outside target {verdict_threshold(documents, False):.2f} z; "
            f"either way the test misses more than half of what sits four "
            f"standard deviations from this reference"
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
            # The most a reference document could deviate, given how many were
            # scored. Carried in the output so a z can be read against what was
            # reachable rather than against 2.5 alone.
            "ceiling": round(ceiling, 6),
            "surprisal_mean": round(calibration["surprisal"]["mean"], 6),
            "surprisal_stdev": round(calibration["surprisal"]["stdev"], 6),
            "novel_ngram_mean": round(calibration["novel_ngram_rate"]["mean"], 6),
            "novel_ngram_stdev": round(calibration["novel_ngram_rate"]["stdev"], 6),
        },
        "documents": scored,
        "warnings": warnings,
    }


# --- Span localization -----------------------------------------------------

def span_surprisals(scores, window):
    """Mean surprisal of every ``window``-token span, in token order."""
    return [
        sum(scores[i:i + window]) / window
        for i in range(len(scores) - window + 1)
    ]


def rank_spans(means, window, top):
    """Indices of the ``top`` highest-scoring spans, without overlaps.

    Neighboring windows share all but one token and so score almost alike; a
    raw ranking returns the same passage a dozen times, one token shifted. Take
    the highest, discard everything it overlaps, repeat.
    """
    kept = []
    for start in sorted(range(len(means)), key=lambda i: (-means[i], i)):
        if all(abs(start - other) >= window for other in kept):
            kept.append(start)
            if len(kept) == top:
                break
    return kept


def explain_span(model, tokens, scores, index, window):
    """Per-token evidence for one span, most surprising token first.

    F7: a flag a reviewer cannot check is not auditable. Each entry names the
    n-gram the model believed and how often it occurs in the reference, so a
    score can be followed back to the counts that produced it.
    """
    order_context = model.order - 1
    entries = []
    for position in range(index, index + window):
        # explain() truncates the context itself, so this slice is about not
        # copying the whole document prefix per token; it matches how
        # NgramModel.surprisals walks the same tokens.
        context = tokens[max(0, position - order_context):position]
        entry = model.explain(context, tokens[position])
        entry["bits"] = round(scores[position], 6)
        entries.append(entry)
    return sorted(entries, key=lambda entry: -entry["bits"])


def describe_span(text, located, starts, index, window, score, explanation):
    """Turn a span index into something a reviewer can look up and read."""
    first_offset = located[index][1]
    last_offset = located[index + window - 1][2]
    return {
        "explanation": explanation,
        "bits_per_token": round(score, 6),
        "first_token": index,
        "first_line": markov_core.line_of(starts, first_offset),
        "last_line": markov_core.line_of(starts, last_offset),
        # The source as written, not as tokenized: a reviewer needs to
        # recognize the passage in the document, and the model's view of it has
        # dropped the punctuation and the capitals.
        "text": " ".join(text[first_offset:last_offset].split()),
    }


def document_spans(model, text, located, scores, window, top):
    """Rank the most and least expected passages of one document.

    ``scores`` is the document's per-token surprisal, already computed for its
    overall score: the spans are a second reading of the same numbers, not a
    second pass over the document.
    """
    # A document shorter than the window yields no spans on its own:
    # span_surprisals has nothing to average over.
    if top < 1:
        return {"window": window, "most_variant": [], "least_variant": []}

    starts = markov_core.line_starts(text)
    means = span_surprisals(scores, window)

    tokens = [token for token, _, _ in located]

    def described(indices):
        return [
            describe_span(
                text,
                located,
                starts,
                i,
                window,
                means[i],
                explain_span(model, tokens, scores, i, window),
            )
            for i in indices
        ]

    highest = rank_spans(means, window, top)
    lowest = rank_spans([-mean for mean in means], window, top)
    return {
        "window": window,
        "most_variant": described(highest),
        "least_variant": described(lowest),
    }


def _span_lines(spans, explain):
    """Render the ranked passages: the report's actual deliverable.

    A document score says something is unusual; these say where to look.
    """
    if not spans["most_variant"]:
        return []

    lines = ["", f"  most variant spans ({spans['window']}-token window):"]
    for span in spans["most_variant"]:
        lines.append(_span_line(span))
        lines.extend(_evidence_lines(span, explain, variant=True))
    lines.append("")
    lines.append("  least variant spans:")
    for span in spans["least_variant"]:
        lines.append(_span_line(span))
        lines.extend(_evidence_lines(span, explain, variant=False))
    return lines


def _evidence_lines(span, explain, variant):
    """The counts behind a span: the token that decided it, or all of them.

    Every reported span carries its evidence by default, because a flag whose
    basis is only in the JSON is a flag most readers will never check.

    Which token decided it depends on the end of the ranking. A variant span is
    driven by its least expected word; a familiar one is held down by its most
    expected, and quoting that span's worst token would explain the wrong
    thing.
    """
    if not span["explanation"]:
        return []

    # explanation is sorted most surprising first.
    entries = span["explanation"] if variant else list(reversed(span["explanation"]))
    if not explain:
        label = "driven by" if variant else "anchored by"
        return [f"           {label} {_evidence(entries[0])}"]
    return [
        f"           {entry['bits']:5.1f}  {_evidence(entry)}" for entry in entries
    ]


def _evidence(entry):
    """State one token's evidence as a fact about the reference corpus."""
    word = entry["word"]
    if entry["backoff_level"] < 0:
        return f'"{word}" never appears in the reference'
    if entry["backoff_level"] == 0:
        offered = " ".join(entry["context_offered"])
        seen = f'"{word}" appears {entry["count"]:,}x'
        return f'{seen}, but never after "{offered}"' if offered else seen
    matched = " ".join(entry["context_used"] + [word])
    return f'"{matched}" appears {entry["count"]:,}x'


def _span_line(span):
    location = (
        f"l.{span['first_line']}"
        if span["first_line"] == span["last_line"]
        else f"ll.{span['first_line']}-{span['last_line']}"
    )
    return f"    {span['bits_per_token']:5.1f}  {location:<12} \"...{span['text']}...\""


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


def format_analysis(analysis, patterns, explain=False):
    """Render an analysis as the human-readable report.

    The reference and its expected range are stated once, above the documents
    scored against them: they are the same for every document in one run, and a
    score is meaningless without them.
    """
    reference = analysis["reference"]
    calibration = analysis["calibration"]
    order = analysis["order"]
    member_threshold = verdict_threshold(calibration["documents"], True)
    outside_threshold = verdict_threshold(calibration["documents"], False)

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
        f"            variant past {member_threshold:.2f} z for a reference "
        f"document, which can reach at most {calibration['ceiling']:.2f};",
        f"            past {outside_threshold:.2f} z for a target from "
        f"outside, which is not bounded",
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
                "model built without it,"
            )
            lines.append(
                f"  called variant past {document['threshold']:.2f} z and bounded at "
                f"{calibration['ceiling']:.2f} by the "
                f"{calibration['documents']} documents it is measured among"
            )
        lines.extend(_span_lines(document["spans"], explain))

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
        reference,
        targets,
        args.tokenizer,
        args.order,
        args.baseline,
        args.window,
        args.top_spans,
    )

    if args.report == "json":
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print(format_analysis(analysis, args.reference, args.explain))

    return 0


# --- generate --------------------------------------------------------------

# Order 2 for generation, not the analysis default of 3. At 128k words an
# order-3 chain is forced at 94% of its states (docs/ROADMAP.md), so it mostly
# transcribes its source. Order 2 is the point where the output is still varied.
GENERATE_ORDER = 2
GENERATE_WORDS = 100
WRAP_COLUMNS = 70

# --sentences needs a stop condition that a corpus might never supply. Cap the
# walk so a corpus without sentence-ending punctuation ends the run and says so
# rather than looping until interrupted.
SENTENCE_TOKEN_LIMIT = 2000

# A sentence ends at .!? even when quotes or brackets close after it. Naive
# about abbreviations: "Mr." ends a sentence as far as this is concerned.
_SENTENCE_END = re.compile(r"""[.!?][)\]"'”’]*$""")


def ends_sentence(token):
    """Whether a token closes a sentence."""
    return bool(_SENTENCE_END.search(token))


def opening_context(tokens, chain, order, rng):
    """Choose where to start the walk, preferring a real sentence opening.

    A capitalized first word is only a proxy for the start of a sentence, and a
    weak one -- "American" opens no sentence in "...the American sound." Taking
    contexts that actually follow sentence-ending punctuation in the corpus
    avoids a leading fragment, which matters most under --sentences, where
    whole sentences are what was asked for.

    Falls back to the capitalized proxy, and then to any context at all. A
    corpus without capitals or sentence punctuation is unusual, not unusable.
    """
    candidates = [
        tuple(tokens[i:i + order])
        for i in range(len(tokens) - order)
        if i == 0 or ends_sentence(tokens[i - 1])
    ]
    if not candidates:
        candidates = markov_core.capitalized_contexts(chain) or list(chain)
    if not candidates:
        raise SystemExit("corpus has no word sequences to generate from")
    return rng.choice(candidates)


def generate_tokens(chain, start, order, rng, words=None, sentences=None):
    """Walk the chain, returning (tokens, note) where note explains a short run.

    ``words`` counts the whole output, the opening context included, so
    ``--words 40`` yields forty words rather than forty plus however many the
    walk was seeded with. A count smaller than the opening context trims it: a
    prefix of a valid walk is still a valid walk, and the alternative is
    ``--words 1`` quietly returning two.

    Two stopping conditions, and a third nobody asked for: the corpus can run
    out. Saying so is what the note is for; looping back to the first word
    instead would splice two unrelated passages together and report nothing.
    """
    start = list(start)

    if sentences is None:
        # The trim below is what enforces the count; this limit only stops the
        # walk doing work that would be discarded, and keeps a count smaller
        # than the opening from reaching the engine as a negative limit.
        produced = list(
            markov_core.walk(
                chain, start, order, rng=rng, limit=max(0, words - len(start))
            )
        )
        out = (start + produced)[:words]
        note = (
            None if len(out) >= words
            else f"the corpus ran out after {len(out):,} words"
        )
        return out, note

    found = []

    def stop(token):
        if ends_sentence(token):
            found.append(token)
        return len(found) >= sentences

    produced = list(
        markov_core.walk(
            chain, start, order, rng=rng, limit=SENTENCE_TOKEN_LIMIT, stop=stop
        )
    )
    note = None
    if len(found) < sentences:
        ran_out = len(produced) < SENTENCE_TOKEN_LIMIT
        note = (
            f"stopped after {len(found)} of {sentences} sentences: "
            + (
                f"the corpus ran out after {len(start) + len(produced):,} words"
                if ran_out
                else f"reached the {SENTENCE_TOKEN_LIMIT:,}-token limit"
            )
        )
    return start + produced, note


def cmd_generate(args):
    documents = read_documents(resolve_reference(args.reference))

    # Whitespace tokenization, always. The plain tokenizer drops the case and
    # punctuation that generated text has to carry to read like its source.
    tokens = markov_core.whitespace_tokens(" ".join(text for _, text in documents))
    if len(tokens) <= args.order:
        raise SystemExit(
            f"corpus of {len(tokens):,} tokens is too short to generate at "
            f"order {args.order}"
        )

    words = args.words
    if words is None and args.sentences is None:
        words = GENERATE_WORDS

    chain = markov_core.build_chain(tokens, args.order)
    rng = random.Random(args.seed)
    start = opening_context(tokens, chain, args.order, rng)
    produced, note = generate_tokens(
        chain, start, args.order, rng, words, args.sentences
    )
    text = textwrap.fill(" ".join(produced), WRAP_COLUMNS)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        # To stderr, so `--output` leaves stdout empty and a pipeline that
        # expects only generated text on stdout is not fed a status line.
        print(f"wrote {len(produced):,} words to {args.output}", file=sys.stderr)
    else:
        print(text)

    if note:
        print(f"note: {note}", file=sys.stderr)
    return 0


# --- Dispatch --------------------------------------------------------------

def positive_int(value):
    """An argparse type for arguments that must be 1 or greater."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {number}")
    return number


def non_negative_int(value):
    """An argparse type for counts where zero is a meaningful 'none'."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"cannot be negative, got {number}")
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
        help="how to calibrate the expected range: leave-one-out scores every "
        "reference document against the others, holdout scores every fifth "
        "(default: %(default)s)",
    )
    analyze.add_argument(
        "--window",
        type=positive_int,
        default=12,
        help="span length in tokens (default: %(default)s)",
    )
    analyze.add_argument(
        "--top-spans",
        type=non_negative_int,
        default=5,
        help="spans to report at each end of the ranking, 0 for none "
        "(default: %(default)s)",
    )
    analyze.add_argument(
        "--explain",
        action="store_true",
        help="show the reference counts behind every token of a reported span, "
        "not only the strongest one",
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

    generate = subcommands.add_parser(
        "generate",
        help="generate imitative text from a corpus",
        description="Walk the same n-gram counts forwards to produce text in "
        "the style of a corpus. Always whitespace-tokenized: generated text "
        "has to carry the case and punctuation that the analysis tokenizer "
        "drops.",
    )
    generate.add_argument(
        "--reference",
        nargs="+",
        required=True,
        metavar="PATH",
        help="corpus to generate from: directories, globs, or files",
    )
    generate.add_argument(
        "--order",
        type=positive_int,
        default=GENERATE_ORDER,
        help="chain order (default: %(default)s). Higher orders quote the "
        "corpus at greater length; see docs/ROADMAP.md",
    )
    length = generate.add_mutually_exclusive_group()
    length.add_argument(
        "--words",
        type=positive_int,
        help=f"how many words to generate (default: {GENERATE_WORDS})",
    )
    length.add_argument(
        "--sentences",
        type=positive_int,
        help="generate whole sentences instead of a word count, stopping on "
        "the punctuation that ends one",
    )
    generate.add_argument(
        "--seed",
        type=int,
        help="seed the generator so a run can be reproduced exactly; "
        "without it, output differs every time",
    )
    generate.add_argument(
        "--output",
        metavar="PATH",
        help="write the text to a file instead of stdout",
    )
    generate.set_defaults(func=cmd_generate)

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

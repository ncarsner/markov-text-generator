# PRD — Language Deviation Analysis

**Status:** proposed · **Branch:** `analysis` · **Depends on:** `uv-tooling`
**Scope:** v1 is domain-general. Domain-specific handling is deferred; see
[Later enhancements](#later-enhancements-out-of-scope-for-v1).

## Summary

The principal application of this tool is **identifying and quantifying language
that is calculably variant from expected values**. Given a reference corpus that
establishes what "expected" means, the tool scores a target document and locates
the spans that deviate.

Text generation is retained as a **side effect** of the same n-gram table, not as
the product. A single CLI exposes both, plus corpus diagnostics.

This inverts the emphasis in [ROADMAP.md](ROADMAP.md), which treated generation
as primary. The data structure was always better suited to analysis: a Markov
table *is* a model of expectation, and generation is merely sampling from it.

## Problem

Reviewers must answer "what in this document is unusual?" against a body of
precedent too large to hold in mind. Existing options are poor:

- **Manual review** — does not scale, and misses subtle drift.
- **Diff tools** — require a specific baseline document; cannot express "unlike
  this corpus of 500 documents."
- **LLM review** — capable, but non-deterministic, unauditable, and cannot show
  *why* a span was flagged. In regulated review this is often disqualifying.

An n-gram model is weaker than a transformer at language understanding, and that
is an acceptable trade. Its advantages are the ones this problem actually
requires: it is **deterministic, fully explainable, auditable, offline, and free
of vendor dependency**. Every flag traces to a specific count in a specific
reference document.

## Users and use cases

v1 is domain-general: it operates on any plain-text corpus. The motivating target
domains below are what the tool is built *toward*; none require v1 to understand
them, and domain-specific handling is deferred.

| User | Question | Reference corpus |
|---|---|---|
| Analyst | Does this document introduce language unlike the existing body? | Prior documents of the same kind |
| Reviewer | Which passages deviate from our standard? | Accepted exemplars |
| Owner | Has our language drifted across revisions? | Prior versions |
| Records analyst | Which corpus does this document most resemble? | Several candidate corpora |

The last row is attribution: the reference that scores a document *lowest* is the
one it most resembles.

## Feasibility evidence

Prototyped against the 54-address inaugural corpus (128k words) before writing
this document. All figures below are measured, not projected.

**Document-level.** Leave-one-out over the reference established an expected
surprisal of **9.01 ± 0.47 bits/token**. Two out-of-domain documents scored
above every in-domain document:

| Document | Bits/token | z |
|---|---:|---:|
| Zachary Taylor, inaugural (in-domain, held out) | 7.81 | −2.54 |
| Harding, inaugural (in-domain, held out) | 9.90 | +1.87 |
| Lincoln, second inaugural (in-domain, held out) | 9.99 | +2.06 |
| JFK, Rice University (out-of-domain) | 10.36 | +2.85 |
| FDR, Day of Infamy (out-of-domain) | 10.34 | +2.79 |

Two independent validations: the method separated out-of-domain text from
in-domain, **and** the in-domain documents it ranked most variant — Lincoln's
second inaugural and Harding's — are both independently well known as atypical
presidential prose. The metric tracks something real.

Separation is nonetheless modest (+2.85 vs an in-domain maximum of +2.06), which
is the central design finding:

**Span-level is where the value is.** Ranking 12-token windows of the Rice
speech against the inaugural model produced a ~4× spread:

```
 17.1 bits   ...atmosphere at speeds of over 25 000 miles per hour causing heat...
 16.4 bits   ...atlas which launched john glenn generating power equivalent to 10 000 automobiles...
 15.6 bits   ...shot is comparable to firing a missile from cape canaveral and dropping...
  ...
  4.5 bits   ...be done and it will be done before the end of this...
  4.8 bits   ...the people of the world than those of the soviet union the...
```

The high-surprisal spans are precisely the novel technical content; the low ones
are boilerplate. **Document scores are a summary statistic; ranked spans are the
deliverable.**

## Goals

- **G1** Quantify how far a document deviates from a reference corpus, on a
  calibrated scale with a stated expected range.
- **G2** Localize deviation to specific spans, ranked by surprisal.
- **G3** Report *why* a span was flagged, traceable to reference counts.
- **G4** Remain standard-library-only, offline, and deterministic.
- **G5** Expose generation from the same engine and CLI.

## Non-goals

- **Not domain-aware.** v1 treats all text as plain prose. It has no concept of
  citations, defined terms, clauses, or document structure.
- **Not legal advice.** The tool measures statistical deviation from a corpus. It
  does not assess legal significance, risk, enforceability, or compliance.
- Not a semantic or entailment model. It does not know what words mean.
- Not a diff tool for two specific documents.
- No ML frameworks, no network calls at analysis time, no telemetry.
- Not a drafting assistant.

## Product surface

One CLI, three subcommands. `analyze` is primary; `generate` is the by-product;
`stats` reports whether a reference corpus is large enough to trust.

```
markov analyze  TARGET... --reference DIR [--order N] [--baseline loo|holdout]
                          [--window N] [--top-spans N] [--report text|json]
markov generate           --reference DIR [--order N] [--words N] [--sentences N] [--seed N]
markov stats              --reference DIR [--order N]
```

### Example output

```
document: speech_we_choose_to_go_to_the_moon.txt        (2,133 tokens)
reference: data/input/speech_inaugural_*.txt            (54 docs, 125,797 tokens)

  surprisal        10.36 bits/token
  expected          9.01 +/- 0.47            leave-one-out, 54 reference docs
  deviation         z = +2.85                VARIANT

  most variant spans (12-token window):
    17.1  ll.61-63   "...atmosphere at speeds of over 25,000 miles per hour, causing heat..."
    16.4  ll.78-80   "...Atlas which launched John Glenn, generating power equivalent to 10,000 automobiles..."
```

## Functional requirements

**F1 — Corpus ingestion.** Load a reference corpus from a directory or file list.
Report token count, vocabulary, and per-document contribution.

**F2 — Model construction.** Build n-gram counts to configurable order (default
3). Order must be a parameter, not a constant.

**F3 — Smoothing.** Unseen n-grams must not produce infinite surprisal. Ship
**stupid backoff** (α = 0.4) for v1: simple, unnormalized, effective, and easy to
explain in an audit. Kneser-Ney is the quality ceiling and is deferred.

**F4 — Calibration.** Compute the expected surprisal distribution from the
reference itself via leave-one-out (default) or a held-out split. All deviation
scores are reported as z against this distribution — this is what makes variance
*calculable* rather than merely observed.

**F5 — Document scoring.** Mean surprisal in bits/token, plus z, plus novel
n-gram rate against its own expected range.

**F6 — Span localization.** Sliding-window mean surprisal, ranked, overlapping
windows suppressed. Window size configurable; report source line numbers.

**F7 — Explanation.** For any flagged span, report the backoff level used and the
reference count that drove the score, so a reviewer can trace the flag.

**F8 — Attribution mode.** Score one target against several reference corpora and
rank them by resulting surprisal.

**F9 — Generation.** Retain generation from the same table: order, word or
sentence count, and optional seed.

**F10 — Output formats.** Human-readable text and JSON. JSON is the integration
surface for review pipelines.

## Technical approach

- **Engine** — `markov_core.py` holds counting, backoff scoring, and sampling.
  `markov_cli.py` becomes the subcommand dispatcher. `generate_text()` is
  preserved as a thin wrapper so the existing 57 tests continue to pass unchanged.
- **Tokenization** — v1 ships one plain-prose tokenizer. It is defined behind a
  **pluggable interface** so domain profiles can be added later without touching
  the engine; that seam is the only concession v1 makes to future domain work.
- **Determinism** — analysis is fully deterministic. Only `generate` draws
  randomness, and `--seed` makes that reproducible too.
- **Performance** — target corpora are 10⁵–10⁷ tokens. Use `Counter` rather than
  duplicate-storing lists (see ROADMAP); scoring is O(tokens × order).

## Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M1 | `markov_core.py`: counts, backoff, sampling — **done** | Existing 57 tests pass unchanged |
| M2 | `stats` subcommand — **done** | Reproduces the ROADMAP branching table |
| M3 | `analyze`: document scoring + calibration | Reproduces the 9.01 ± 0.47 baseline |
| M4 | Span localization and ranking | Reproduces the prototype's span ranking |
| M5 | Explanation trace (F7) + JSON output | Every flag traceable to a reference count |
| M6 | `generate` folded into the CLI | `markov.py` keeps zero-arg behavior; `--output` restores the file export dropped with the positional interface at M2 |

## Acceptance criteria

- **A1** On a held-out reference document, `|z| < 2.5` for at least 90% of the
  corpus.
- **A2** An out-of-domain document of the same era and register scores `z > +2.5`.
- **A3** Span ranking places novel content above boilerplate, as in the prototype.
- **A4** Every flagged span reports the reference count behind it.
- **A5** Analysis output is byte-identical across runs on identical input.
- **A6** No third-party runtime dependency; no network access during analysis.
- **A7** Test coverage matches the existing standard, including mutation checks.

## Risks and limitations

| Risk | Mitigation |
|---|---|
| **Small reference corpus gives unreliable scores.** Demonstrated: at 2,648 words the model was 92% forced and near-useless. | `stats` gates analysis; warn below a token threshold. |
| **Misread as judgment rather than measurement.** A statistical outlier is not necessarily a problem. | Non-goals stated in output header, not only in docs. |
| **Reference corpus encodes its own bias.** "Expected" means "typical of what you supplied." | Report corpus composition alongside every score. |
| **Tokenization artifacts.** Numbers and rare proper nouns are inherently novel and may crowd span rankings. | Accepted in v1; measure the effect and use it to specify domain profiles. |
| **n-gram ceiling.** Cannot detect semantic deviation expressed in familiar wording. | Stated plainly; the trade is auditability for capability. |
| **Genre confound.** A document may score variant for register rather than substance. | Attribution mode (F8) helps distinguish. |

## Later enhancements (out of scope for v1)

Deferred deliberately so v1 can be validated end-to-end on a general corpus
first. Each would be specified in its own PRD.

### Domain profile: legal and policy text

The motivating domain — legislation, contracts, policy language — but it needs
handling v1 does not provide, and specifying it now would block validation of the
core engine:

- **Citation normalization.** Strings like `17 U.S.C. § 105` are inherently novel
  and would otherwise dominate every span ranking as permanent noise.
- **Defined terms.** `"Confidential Information"` is multiword and
  case-significant; a plain lowercasing tokenizer destroys the signal that a term
  is defined rather than incidental.
- **Structural segmentation.** Sections, subsections, and enumerations are
  structure, not vocabulary, and should segment spans rather than appear in them.
- **Clause-aligned spans.** Fixed windows cut across clause boundaries; legal
  review wants the clause as the unit.
- **Corpus acquisition.** Candidates are US Code, the Federal Register, and SEC
  EDGAR filings — all public record.

The pluggable tokenizer seam in v1 is what makes this additive rather than a
rewrite.

### Other deferred work

- **Kneser-Ney smoothing** — the quality ceiling above stupid backoff.
- **Percentile scoring** instead of z, which is more robust to non-normality.
- **Drift over time** — scoring a document series against a rolling reference.

## Open questions

1. **Repository name.** `markov-text-generator` describes the by-product, not the
   principal application. Rename?
2. **Is z the right scale for v1**, or should output be a percentile against the
   reference distribution?
3. **Does `markov.py` remain** as the zero-argument demo front door once `markov
   generate` exists?

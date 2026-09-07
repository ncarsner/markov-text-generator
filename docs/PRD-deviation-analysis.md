# PRD — Language Deviation Analysis

**Status:** proposed · **Branch:** `analysis` · **Depends on:** `uv-tooling`

## Summary

The principal application of this tool is **identifying and quantifying language
that is calculably variant from expected values**. Given a reference corpus that
establishes what "expected" means for a domain, the tool scores a target
document and locates the spans that deviate.

Text generation is retained as a **side effect** of the same n-gram table, not as
the product. A single CLI exposes both.

This inverts the emphasis in [ROADMAP.md](ROADMAP.md), which treated generation
as primary. The data structure was always better suited to analysis: a Markov
table *is* a model of expectation, and generation is merely sampling from it.

## Problem

Reviewers of legal and policy text must answer "what in this document is
unusual?" against a body of precedent too large to hold in mind. Existing options
are poor:

- **Manual review** — does not scale, and misses subtle drift.
- **Diff tools** — require a specific baseline document; cannot express "unlike
  this corpus of 500 contracts."
- **LLM review** — capable, but non-deterministic, unauditable, and cannot show
  *why* a clause was flagged. In regulated review this is often disqualifying.

An n-gram model is weaker than a transformer at language understanding, and that
is an acceptable trade. Its advantages are the ones this domain actually
requires: it is **deterministic, fully explainable, auditable, offline, and free
of vendor dependency**. Every flag traces to a specific count in a specific
reference document.

## Users and use cases

| User | Question | Reference corpus |
|---|---|---|
| Legislative analyst | Does this amendment introduce language unlike existing code? | Current statute / prior sessions |
| Contract reviewer | Which clauses deviate from our standard template? | Executed agreements of the same type |
| Policy owner | Has our policy language drifted across revisions? | Prior policy versions |
| Compliance | Does this filing read unlike peer filings? | Peer/industry filings |
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
are political boilerplate. **Document scores are a summary statistic; ranked
spans are the deliverable.**

## Goals

- **G1** Quantify how far a document deviates from a reference corpus, on a
  calibrated scale with a stated expected range.
- **G2** Localize deviation to specific spans, ranked by surprisal.
- **G3** Report *why* a span was flagged, traceable to reference counts.
- **G4** Remain standard-library-only, offline, and deterministic.
- **G5** Expose generation from the same engine and CLI.

## Non-goals

- **Not legal advice.** The tool measures statistical deviation from a corpus. It
  does not assess legal significance, risk, enforceability, or compliance. A
  flagged span may be unremarkable; an unflagged one may be critical.
- Not a semantic or entailment model. It does not know what words mean.
- Not a diff tool for two specific documents.
- No ML frameworks, no network calls at analysis time, no telemetry.
- Not a drafting assistant.

## Product surface

One CLI, three subcommands. `analyze` is primary; `generate` is the by-product.

```
markov analyze  TARGET... --reference DIR [--order N] [--baseline loo|holdout]
                          [--window N] [--top-spans N] [--report text|json]
markov generate           --reference DIR [--order N] [--words N] [--sentences N] [--seed N]
markov stats              --reference DIR [--order N]
```

`stats` reports corpus diagnostics — vocabulary, branching factor, forced-state
share — which determine whether a reference corpus is large enough to trust.

### Example output

```
document: proposed_amendment_14.txt          (2,118 tokens)

  surprisal        11.42 bits/token
  expected          9.01 +/- 0.47            leave-one-out, 54 reference docs
  deviation         z = +5.13                HIGHLY VARIANT
  novel trigrams      41%                    expected 22% +/- 4%

  most variant spans (12-token window):
    17.4  §3(b) ll.42-44   "...notwithstanding any provision to the contrary in subchapter..."
    16.1  §7(a) ll.98-101  "...algorithmic determination of eligibility shall be subject to..."
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
- **Tokenization** is the highest-risk design decision for legal text, and needs
  domain handling that the current `str.split()` does not provide:
  - Citations (`17 U.S.C. § 105`) will otherwise register as permanent
    high-surprisal noise; normalize to a placeholder token.
  - Defined terms (`"Confidential Information"`) are multiword and
    case-significant; naive lowercasing destroys the signal that they are defined.
  - Section numbers and enumerations should be structural, not lexical.
  - Decision: pluggable tokenizer, with a `legal` profile alongside `plain`.
- **Determinism** — analysis is fully deterministic. Only `generate` draws
  randomness, and `--seed` makes that reproducible too.
- **Performance** — target corpora are 10⁵–10⁷ tokens. Use `Counter` rather than
  duplicate-storing lists (see ROADMAP); scoring is O(tokens × order).

## Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M1 | `markov_core.py`: counts, backoff, sampling | Existing 57 tests pass unchanged |
| M2 | `stats` subcommand | Reproduces the ROADMAP branching table |
| M3 | `analyze`: document scoring + calibration | Reproduces the 9.01 ± 0.47 baseline |
| M4 | Span localization and ranking | Reproduces the prototype's span ranking |
| M5 | Explanation trace (F7) + JSON output | Every flag traceable to a reference count |
| M6 | `generate` folded into the CLI | `markov.py` keeps zero-arg behavior |
| M7 | Legal tokenizer profile | Citations no longer dominate span rankings |

## Acceptance criteria

- **A1** On a held-out reference document, `|z| < 2.5` for at least 90% of the
  corpus.
- **A2** An out-of-domain document of the same era and register scores `z > +2.5`.
- **A3** Span ranking places novel technical content above boilerplate, as in the
  prototype.
- **A4** Every flagged span reports the reference count behind it.
- **A5** Analysis output is byte-identical across runs on identical input.
- **A6** No third-party runtime dependency; no network access during analysis.
- **A7** Test coverage matches the existing standard, including mutation checks.

## Risks and limitations

| Risk | Mitigation |
|---|---|
| **Small reference corpus gives unreliable scores.** Demonstrated: at 2,648 words the model was 92% forced and near-useless. | `stats` gates analysis; warn below a token threshold. |
| **Misread as legal judgment.** A statistical outlier is not a legal problem. | Non-goals stated in output header, not only in docs. |
| **Reference corpus encodes its own bias.** "Expected" means "typical of what you supplied." | Report corpus composition alongside every score. |
| **Tokenization artifacts dominate.** Citations and numbers are inherently novel. | Legal tokenizer profile (M7); normalize before scoring. |
| **n-gram ceiling.** Cannot detect semantic deviation with familiar wording. | Stated plainly; the trade is auditability for capability. |
| **Genre confound.** A document may score variant for register rather than substance. | Attribution mode (F8) helps distinguish. |

## Open questions

1. **Repository name.** `markov-text-generator` describes the by-product, not the
   principal application. Rename?
2. **Corpus acquisition for legal text.** The current corpus is presidential
   speeches — a good testbed, wrong domain. Candidates: US Code, Federal
   Register, SEC EDGAR filings. All public domain or public record.
3. **Is z the right scale**, or should output be a percentile against the
   reference distribution, which is more robust to non-normality?
4. **Sentence- or clause-level segmentation** instead of fixed windows, so spans
   align with legal structure?
5. **Does `markov.py` remain** as the zero-argument demo front door once `markov
   generate` exists?

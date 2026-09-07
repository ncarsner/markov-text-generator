# markov-text-generator

Measures how far a document's language deviates from what a reference corpus
leads you to expect — and, as a side effect of the same machinery, generates
imitative text from that corpus.

## What this does

Give it a body of text you consider representative (the **reference corpus**).
It builds a statistical model of the word sequences that corpus contains, then
scores a **target document** against it, answering two questions:

1. **How unusual is this document overall?**
2. **Which specific passages are the unusual ones?**

The second is the useful one. A document-level score tells you something is off;
a ranked list of passages tells you where to look.

Because the model is nothing more than counted word sequences, every score
traces back to specific counts in specific reference documents. It is
deterministic, fully explainable, runs offline, and has no dependencies beyond
the Python standard library. It is far weaker than a language model at
understanding text — that is the trade, and for review work where you must be
able to justify a flag, it is often the right one.

See [docs/PRD-deviation-analysis.md](docs/PRD-deviation-analysis.md) for the full
specification and [docs/ROADMAP.md](docs/ROADMAP.md) for engineering notes.

## Status

Under active development. The engine is built; the CLI that exposes it is not
finished yet.

| | State |
|---|---|
| `markov_core.py` — counts, backoff scoring, sampling | **working** |
| `markov analyze` — document + span scoring | planned |
| `markov stats` — corpus diagnostics | planned |
| `markov generate` — text generation | planned |
| `markov_cli.py` — the current, older generation script | **working** |

Everything in [Understanding the metrics](#understanding-the-metrics) works today
through `markov_core`; the examples below are runnable.

## Understanding the metrics

Six numbers, in plain language. **You do not need statistics to read these.**

---

### Surprisal — "bits per token"

**In plain terms.** How unexpected the wording is. Low means the document reads
like the reference corpus. High means it does not.

**Precisely.** For each word, the model asks how likely that word was given the
two words before it, and converts the answer to bits. The document's score is
the average across all its words.

**How to read it.** The scale is logarithmic: **each extra bit means twice as
unexpected.** A document at 10 bits is roughly twice as surprising as one at 9.
There is no universal good or bad value — see *the expected range* below.

---

### Perplexity

**In plain terms.** The same measurement, in a friendlier unit: roughly *how many
different words the model was choosing between* at each step.

**Precisely.** `2 ^ bits-per-token`.

**How to read it.** Our 54-speech reference corpus scores 9.01 bits, which is a
perplexity of about **516** — at each word the model was effectively picking from
around 516 options. Lower means more predictable, more formulaic text.

---

### The expected range

**In plain terms.** What counts as "normal" for *your* corpus. This is the number
that makes deviation calculable rather than merely observed.

**Precisely.** Each reference document is scored against a model built from all
the *others* (leave-one-out). The mean and standard deviation of those scores
become the expected range.

**How to read it.** Our corpus expects **9.01 ± 0.47 bits per token**. That range
is a property of that corpus alone. A different corpus has a different range, and
scores are never comparable across corpora.

---

### Deviation — the z score

**In plain terms.** How far outside normal the document falls, counted in
standard deviations.

**Precisely.** `(document score − expected mean) ÷ expected standard deviation`.

**How to read it.**

| z | Reading |
|---|---|
| between −1 and +1 | Typical. Unremarkable for this corpus. |
| ±1 to ±2 | Noticeably different, but within ordinary variation. |
| beyond ±2.5 | Variant. Worth a look. |

Negative is not "good" — it means *more formulaic than typical*, which can be its
own signal (heavy boilerplate, copied language).

---

### Novel n-gram rate

**In plain terms.** The share of word sequences in the document that never appear
in the reference at all.

**How to read it.** The most directly interpretable number here — no scale to
learn. In the example below, 86% of the moon speech's three-word sequences are
absent from the inaugural corpus.

---

### Branching factor and forced states

**In plain terms.** Whether your reference corpus is big enough to trust.

**Precisely.** *Branching* is the average number of different words that follow a
given context. *Forced* is the share of contexts with exactly one possible
continuation — where the model has no choice at all.

**How to read it.** A corpus that is mostly forced has memorized itself rather
than learned a pattern, and makes both a poor reference and a poor generator.
**Check this before trusting any score.**

```
corpus of 125,797 words:   order 1: branching 6.88, forced 45%
                           order 2: branching 1.75, forced 78%
                           order 3: branching 1.14, forced 93%
```

Higher order means more context and better judgment, but more forced states.
A corpus of a few thousand words is over 90% forced at order 2 and is not usable.

---

### Reading a span ranking

The actionable output. Each passage is scored on its own, and the extremes are
what matter:

```
 17.1 bits   ...atmosphere at speeds of over 25,000 miles per hour, causing heat...
 16.4 bits   ...Atlas which launched John Glenn, generating power equivalent to 10,000 automobiles...
  ...
  4.5 bits   ...be done, and it will be done before the end of this...
```

The high-surprisal passages are the novel technical content. The low ones are
boilerplate the corpus has seen many times.

---

### What these numbers do **not** mean

A high score means **unlike the reference corpus**. It does not mean wrong,
poorly written, incorrect, risky, or non-compliant. Novel-but-fine language and
genuinely anomalous language score the same. The tool narrows where a human
looks; it does not decide anything.

Scores also depend on the corpus you chose. "Expected" only ever means "typical
of what you supplied," so a biased or unrepresentative reference produces
confident, meaningless numbers.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                    # create the environment
uv run python scripts/fetch_corpus.py      # download the sample corpus
uv run pytest                              # 126 tests
```

Score a document against a reference corpus:

```python
import glob
import markov_core as mc

reference = " ".join(open(f, encoding="utf-8").read()
                     for f in glob.glob("data/input/speech_inaugural_*.txt"))
model = mc.NgramModel(mc.plain_tokens(reference), order=3)

target = mc.plain_tokens(open("data/input/speech_we_choose_to_go_to_the_moon.txt",
                              encoding="utf-8").read())

print(f"{model.bits_per_token(target):.2f} bits/token")   # 10.36
print(f"{model.novel_ngram_rate(target):.0%} novel")      # 86%
```

Against the expected range of 9.01 ± 0.47, that is z = +2.85 — variant, as you
would hope, since a speech about spaceflight is not an inaugural address.

Generate text (the existing script):

```sh
uv run python markov_cli.py data/input/speech_day_of_infamy.txt 50
```

## The corpus

`data/` is intentionally **not** tracked by git, so no text is redistributed
here. `scripts/fetch_corpus.py` rebuilds it from
[Project Gutenberg ebook #925](https://www.gutenberg.org/ebooks/925) — 54
presidential inaugural addresses, 1789–2005.

Every text used is a work of the United States federal government and is
uncopyrighted under
[17 U.S.C. § 105](https://www.law.cornell.edu/uscode/text/17/105). This is
deliberate: the generator reproduces multi-word spans from its source verbatim,
so generated output inherits the licence of the corpus it was trained on.

## Two views of the same text

Tokenizer choice changes every number, so it is stated with each result:

- **`plain_tokens`** — lowercased, punctuation dropped. Used for analysis, where
  `Freedom!` and `freedom` are the same evidence.
- **`whitespace_tokens`** — case and punctuation preserved. Used for generation,
  where output should read like the source.

## Development

```sh
uv run pytest              # full suite
uv run pytest -v           # per-test names
```

Tests are checked against deliberately injected bugs, not just written to pass —
if you add behavior, add a test and confirm it fails when the behavior is broken.

## Limitations

- Counts word sequences; has no idea what words mean. Semantic deviation
  expressed in ordinary wording is invisible to it.
- Needs a reference corpus of roughly 10⁵ words or more. Below that, scores are
  dominated by sparsity.
- Not a diff tool, not a compliance system, and not a substitute for review.

## Credit

Forked from [Ben Hoyt's](https://benhoyt.com/writings/markov-chain/) Markov chain
implementation, which is where the generator began.

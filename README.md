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

Under active development. The engine is built and the CLI is being filled in one
subcommand at a time.

| | State |
|---|---|
| `markov_core.py` — counts, backoff scoring, sampling | **working** |
| `markov_cli.py stats` — corpus diagnostics | **working** |
| `markov_cli.py analyze` — document scoring | **working** |
| `markov_cli.py analyze` — span localization | planned |
| `markov_cli.py generate` — text generation | planned |
| `markov.py` — the original generation script | **working** |

Everything in [Understanding the metrics](#understanding-the-metrics) works today
through `markov_core` or `stats`; the examples below are runnable. Until
`generate` lands, `markov.py` remains the way to generate text.

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

**How it is built.** A score of 10.36 bits means nothing on its own — there is no
universal scale. So the corpus is asked what it expects of its own members:

> Take reference document #1 out. Build the model from the other 53. Score
> document #1 against it. Put it back, take out #2, and repeat — 54 times, so
> every document gets scored by a model that never saw it.

Those 54 scores are what "normal" means for this corpus. Their average is 9.01
and they spread about 0.47 either side, so the corpus expects **9.01 ± 0.47 bits
per token**. Statistics calls this *leave-one-out*; the tool prints the procedure
rather than the term.

The reason a document is held out is that a model which has already read it would
recognize its exact wording and call it unsurprising. It would be grading its own
homework. The same rule applies when you score a document that is part of your
reference corpus: the tool notices, drops it from the model, and says so.

`--baseline holdout` does the cheap version — hold out every fifth document once,
build one model instead of 54. On the inaugural corpus that is 0.2 seconds against
5, at the cost of a coarser estimate from fewer scores.

**How to read it.** That range is a property of that corpus alone. A different
corpus has a different range, and scores are never comparable across corpora.

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

### The ranking

**In plain terms.** The same finding without any statistics: how many reference
documents this one out-scores.

> `reads as more unusual than 54 of the 54 reference documents`

**How to read it.** This is the line to quote to someone who does not want a
z score, but it is blunter than it looks. Harding's inaugural reads as more
unusual than 52 of 54 and is still ordinary at z = +1.87, and once a document
passes every reference document the count stops distinguishing — the moon speech
and the Day of Infamy speech both sit at 54 of 54, while z still separates them
(+2.85 and +2.79). Lead with the count, decide on the z.

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

`markov_cli.py stats` reports exactly this for your own corpus. Measured on the
56-document public-domain corpus. **Both tokenizers are shown because the choice
changes every number** — this is the same corpus twice, not a disagreement:

| Tokenizer | Tokens | Vocab | order 1 | order 2 | order 3 |
|---|---:|---:|---|---|---|
| `plain` (analysis) | 129,158 | 9,077 | 6.88 / 45% | 1.75 / 78% | 1.14 / 93% |
| `whitespace` (generation) | 128,445 | 14,715 | 4.80 / 57% | 1.59 / 82% | 1.11 / 94% |

`plain` lowercases and drops punctuation, so it has a smaller vocabulary and more
evidence per state — which is why it always looks less forced than `whitespace`
on identical text. [ROADMAP.md](docs/ROADMAP.md) reports the `whitespace` row,
because it is concerned with the generator.

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

It also says nothing about **who or what wrote a document**. A passage that
deviates from an author's previous work deviates for some reason, and a different
author is only one of them — a new subject, a new format, an editor, a co-writer,
a decade's gap, or simply a better day at the desk all move the number the same
way. The tool measures distance from a corpus. It cannot measure cause, and it
cannot attribute authorship.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                    # create the environment
uv run python scripts/fetch_corpus.py      # download the sample corpus
uv run pytest                              # 183 tests
```

Check whether a reference corpus is big enough to score against:

```sh
uv run python markov_cli.py stats \
    --reference 'data/input/speech_inaugural_*.txt' \
    --order 2
```

```
reference: data/input/speech_inaugural_*.txt
           54 documents, 126,488 tokens, 8,898 distinct (plain tokenizer)

  order   branching   forced
      1        6.88     45%
      2        1.75     78%

  branching: distinct words that can follow a context, averaged
  forced:    share of contexts with only one continuation
```

`--tokenizer whitespace` gives the generator's view, `--report json` gives the
same numbers for a pipeline, and a corpus below roughly 10⁵ tokens is flagged as
too sparse to trust. **Check this before believing any score below.**

Score a document against a reference corpus:

```sh
uv run python markov_cli.py analyze \
    data/input/speech_we_choose_to_go_to_the_moon.txt \
    --reference 'data/input/speech_inaugural_*.txt'
```

```
reference:  data/input/speech_inaugural_*.txt
            54 documents, 126,488 tokens, plain tokenizer, order 3
expected:   9.01 +/- 0.47 bits/token, 79% +/- 4% novel 3-grams
            measured by scoring each of the 54 reference documents against the other 53

document:   data/input/speech_we_choose_to_go_to_the_moon.txt  (2,152 tokens)
  surprisal         10.36  bits/token
  deviation         +2.85  z            VARIANT
  novel 3-grams       86%               +1.58 z
  reads as more unusual than 54 of the 54 reference documents

  variant means unlike this reference corpus. It does not mean wrong,
  poorly written, risky, or non-compliant, and "expected" only ever
  means typical of the corpus you supplied.
```

A speech about spaceflight is not an inaugural address, and the numbers say so.
Pass several documents to score them all against one reference, `--baseline
holdout` to trade precision for speed, and `--report json` for a pipeline.

The same measurement through the engine directly:

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


Generate text (the original script, until `markov_cli.py generate` lands):

```sh
uv run python markov.py data/input/speech_day_of_infamy.txt 50
```

## The corpus

`data/` is intentionally **not** tracked by git, so no text is redistributed
here. `scripts/fetch_corpus.py` rebuilds it from
[Project Gutenberg ebook #925](https://www.gutenberg.org/ebooks/925) — 54
presidential inaugural addresses, 1789–2005.

The public-domain corpus is `speech_inaugural_*.txt` plus
`speech_day_of_infamy.txt` and `speech_we_choose_to_go_to_the_moon.txt`. A
working copy of `data/input/` may also hold other material, so **glob
deliberately rather than reaching for `*.txt`** when reproducing published
figures.

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

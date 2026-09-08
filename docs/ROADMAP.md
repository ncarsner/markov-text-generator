# Roadmap and Evaluation

An assessment of where `markov_cli.py` can go using only the standard library,
and where a second-order Markov chain still earns its place in 2026.

> **Superseded in emphasis.** This document treats generation as the primary
> application. The principal application is language deviation analysis; see
> [PRD-deviation-analysis.md](PRD-deviation-analysis.md). The standard-library
> work below remains valid and feeds directly into that effort.

## Corpus policy

Evaluation uses **public-domain sources only**. Every text below is a work of
the United States federal government and is uncopyrighted under
[17 U.S.C. § 105](https://www.law.cornell.edu/uscode/text/17/105):

| Source | Files | Words |
|---|---:|---:|
| F.D. Roosevelt, "Day of Infamy", 8 Dec 1941 | 1 | 515 |
| J.F. Kennedy, Rice University, 12 Sep 1962 | 1 | 2,133 |
| Presidential inaugural addresses, 1789–2005 | 54 | 125,797 |
| **Total** | **56** | **128,445** |

The inaugural addresses come from [Project Gutenberg ebook #925](https://www.gutenberg.org/ebooks/925).
Gutenberg's own boilerplate is separately licensed, so only the text between the
`*** START` / `*** END` markers is used, and three further classes of non-speech
text are stripped:

- `[Transcriber's note: …]` blocks — modern editorial additions, not speech
- place/date header lines beneath each title
- one stray Project Gutenberg attribution line in the 1993 Clinton address

`data/` remains untracked (the blanket `*.txt` rule in `.gitignore`), so no
corpus is redistributed by this repository. Rebuild it from upstream with:

```sh
uv run python scripts/fetch_corpus.py
```

The script applies exactly the cleaning rules above and reproduces the 54 files
used for the measurements below byte-for-byte. It only touches files matching
`speech_inaugural_*`, so the two hand-added speeches are left alone. That matters more than it first
appears — see [Memorization](#memorization-is-the-central-constraint).

## Baseline measurements

Average distinct continuations per state ("branching"), and the share of states
with exactly one possible next word ("forced" — the chain has no choice and must
transcribe the source):

Measured with whitespace tokenization, which is the generator's view of the
text. The analysis engine lowercases and drops punctuation, which yields
different figures for the same corpus; see the README's side-by-side table.
Reproduce this row with `markov_cli.py stats --tokenizer whitespace`.

| Corpus | Words | Vocab | order 1 | order 2 | order 3 |
|---|---:|---:|---|---|---|
| Two speeches only | 2,648 | 1,057 | 2.13 / 76% | 1.14 / 92% | 1.02 / 98% |
| **All 56 speeches** | **128,445** | **14,715** | **4.80 / 57%** | **1.59 / 82%** | **1.11 / 94%** |

Enlarging the corpus roughly 48× lifted order-2 branching from 1.14 to 1.59 and
cut forced states from 92% to 82%. This confirms the prediction below rather
than merely asserting it: corpus size, not algorithm, was the binding
constraint.

### Memorization is the central constraint

Measured as the fraction of a 60-word output that is one contiguous verbatim
lift from the source, over 200 seeds:

| Corpus | Median | ≥50% verbatim | =100% verbatim |
|---|---:|---:|---:|
| Rice University speech alone (2,133 words) | 40% | 46 / 200 | 2 / 200 |
| Two speeches (2,648 words) | 35% | 42 / 200 | 0 / 200 |
| **All 56 speeches (128,445 words)** | **16%** | **1 / 200** | **0 / 200** |

The enlarged corpus removes the worst behavior outright: pure transcription no
longer occurs, and near-total quotation fell from roughly one run in five to one
in two hundred.

It does not eliminate the constraint. At order 2 the chain is still forced at
82% of states, and a median 16% of each output — around nine consecutive words —
is lifted intact. Two consequences survive:

1. **Corpus scale remains the highest-leverage input.** It is now demonstrated,
   not assumed: every further quality gain is cheaper to buy with more text than
   with better code.
2. **Output inherits the corpus's licence.** Because the model still reproduces
   multi-word spans verbatim, generated text from a copyrighted corpus is a
   derivative of it in the most literal sense. This is why the corpus stays
   untracked and why evaluation is restricted to public-domain sources.

## Standard-library evolution

No third-party runtime dependency is needed for any of the following.

### Tier 1 — prerequisites

Small changes that everything else depends on.

- ~~**`argparse`**~~ — **done** for `markov_cli.py`, which is now a subcommand
  dispatcher with `--help` and type validation. The positional-only interface
  and its `0`-means-default sentinel are gone; `markov.py` still carries the
  original positional form until `generate` lands.
- **`random.Random(seed)` + `--seed`** — instantiate a generator rather than
  using the global `random` module. Makes runs reproducible and lets the test
  suite drop its autouse global-seeding fixture, which is currently a smell.
- **`encoding="utf-8"` on every `open()` call** — a live bug, not a feature. A
  corpus containing smart quotes or em-dashes fails on a machine whose locale
  default is not UTF-8. Fixed in `markov_cli.py`, and pinned by a test that
  re-runs the CLI under `-X warn_default_encoding`, where an implicit `open()`
  is an error. `markov.py` still reads with the locale default.

### Tier 2 — capability

- **`--order N` via `collections.deque(maxlen=N)`** — order is baked into the
  `(w1, w2)` tuple today. A bounded deque generalizes it in roughly four lines.
  Given the table above, this converts a fixed, badly-chosen constant into the
  parameter that most needs tuning per corpus.
- **`collections.Counter` + `random.choices(pop, weights=…)`** — `possibles`
  stores a list *with duplicates*, so `random.choice` is already
  frequency-weighted (correct), but a word following a prefix 50 times is stored
  50 times. A `Counter` yields an identical distribution at a fraction of the
  memory. At 2,648 words this was theoretical; at 128,445 it is the difference
  that makes a still-larger corpus practical.
- **`fileinput` + `nargs="+"`** — multiple input files and stdin for free.
  `cat corpus/*.txt | markov --order 3` makes the tool composable.
- **Sentence-aware stopping** — output currently ends mid-clause. Stopping on a
  token ending in `.!?`, exposed as `--sentences N` alongside `--words N`, is a
  handful of lines and the largest perceived-quality gain available.

### Tier 3 — worthwhile, less urgent

- ~~**`--stats`**~~ — **done**, as the `stats` subcommand: it emits the
  branching/forced table for the user's own corpus, in either tokenizer, as text
  or JSON. This is what turns the repository from a script into an instrument;
  see [Teaching](#teaching-strongest-fit).
- **Model persistence** — build once, generate many. Use `json` with encoded
  tuple keys, **not `pickle`**: unpickling executes arbitrary code, and a saved
  model is exactly the sort of file people pass around.
- **`re` tokenization** — `r"\w+|[^\w\s]"` separates punctuation from words so
  `last!` and `last` stop being distinct tokens. Requires a detokenizer to
  reassemble output, so this is a genuine trade-off rather than a clear win.
- **`[project.scripts]` entry point** — `markov = "markov_cli:main"` now that
  `pyproject.toml` exists, giving `uv run markov`.

### Known defects

Both are pinned by the test suite as current-behavior tests, and both are
fixable with the standard library alone.

- **The corpus loops.** The `possibles[w2, ""]` sentinel makes the walk emit
  empty strings at the end of the source and then restart from the first word.
  `split()` and `textwrap.fill()` both collapse the empties, so printed output
  silently splices the corpus end onto its beginning.
- **All-lowercase input crashes.** Seeding requires a capitalized prefix;
  `random.choice` raises `IndexError` on an empty candidate list. Empty input,
  blank lines, and whitespace-only input all reach the same path.

## Where this tool still earns its place

For "produce plausible text", a Markov chain has been obsolete since large
language models. Every use below is one where an LLM is the *wrong* tool rather
than a stronger one.

### Teaching (strongest fit)

The canonical first generative model: ~66 lines, no dependencies, no GPU, fully
inspectable. Adding `--order` and `--stats` makes it a laboratory in which a
student watches order 1 → 4 slide from gibberish into verbatim plagiarism *and
reads the numbers explaining why*. That is a lesson about memorization and
overfitting that transfers directly to reasoning about modern models — and this
repository is already most of the way there.

### Deterministic test fixtures

Seeded, offline, zero-dependency filler carrying a domain's real vocabulary.
Preferable to lorem ipsum for search-index tests, UI overflow tests, and demo
data; unlike an LLM it is instant, free, and reproducible in CI.

### Fuzzing corpora

Structurally plausible but novel inputs for a parser or validator. Markov output
resembles real data while being new, which is what a seed corpus needs.

### Text analysis — the model run backwards (now the principal application)

The same n-gram table answers "how predictable is this document?" Branching
factor and perplexity are real signals for readability scoring, authorship
comparison, and boilerplate detection. Arguably a better fit for the data
structure than generation is — and on that basis this has been promoted from one
use case among six to the tool's stated purpose. See
[PRD-deviation-analysis.md](PRD-deviation-analysis.md).

### Generative art

Cut-up technique, poetry, bot accounts. The artifacts *are* the aesthetic; LLM
output is fluent, which for this purpose is the defect.

### Constrained environments

Air-gapped, embedded, or offline. Runs anywhere Python runs, with no weights and
no network.

### Not suitable for

Production copy, chat, summarization, or translation. An LLM is better on every
axis that matters there.

## Suggested order of work

1. Tier 1 in one pass — `argparse`, `Random(seed)`, UTF-8. Unblocks everything.
   (Corpus scale, previously first here, is now done: 2.6k → 128k words.)
2. `--order` and `--stats` together. They are the teaching story, and `--stats`
   makes the effect of `--order` legible.
3. Fix the two known defects, now that flags exist to control the alternatives.
4. `Counter` + `random.choices`, then multi-file input — the pair that makes a
   corpus large enough to matter practical.
5. Sentence-aware stopping and the entry point as polish.

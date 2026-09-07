# Roadmap and Evaluation

An assessment of where `markov_cli.py` can go using only the standard library,
and where a second-order Markov chain still earns its place in 2026.

## Corpus policy

Evaluation uses **public-domain sources only**. The two speeches below are works
of the United States federal government and are uncopyrighted under
[17 U.S.C. § 105](https://www.law.cornell.edu/uscode/text/17/105):

| File | Source | Words |
|---|---|---|
| `speech_day_of_infamy.txt` | F.D. Roosevelt, 8 Dec 1941 | 515 |
| `speech_we_choose_to_go_to_the_moon.txt` | J.F. Kennedy, Rice University, 12 Sep 1962 | 2,133 |

`data/` remains untracked (the blanket `*.txt` rule in `.gitignore`), so no
corpus is redistributed by this repository. That matters more than it first
appears — see [Memorization](#memorization-is-the-central-constraint).

## Baseline measurements

Average distinct continuations per state ("branching"), and the share of states
with exactly one possible next word ("forced" — the chain has no choice and must
transcribe the source):

| Corpus | Words | Vocab | order 1 | order 2 | order 3 | order 4 |
|---|---:|---:|---|---|---|---|
| Day of Infamy | 515 | 279 | 1.67 / 79% | 1.08 / 95% | 1.02 / 99% | 1.01 / 100% |
| Rice University | 2,133 | 872 | 2.10 / 76% | 1.14 / 92% | 1.02 / 98% | 1.00 / 100% |
| Both combined | 2,648 | 1,057 | 2.13 / 76% | 1.14 / 92% | 1.02 / 98% | 1.00 / 100% |

At the currently hardcoded order 2, **92% of states are forced**. The chain
branches only at the remaining 8%.

### Memorization is the central constraint

Measured on the Rice University speech, 60-word outputs across 200 seeds, as a
fraction of the output that is one contiguous verbatim lift from the source:

- median **40%**
- **46 / 200** runs are at least half a single verbatim quote
- **2 / 200** runs are 100% verbatim — pure transcription with nothing generated

This is the governing fact about the tool. At these corpus sizes it is closer to
a quotation shuffler than a generator. The cause is corpus size, not a code
defect: an order-2 chain wants 10^5+ words and has 10^3.

Two consequences:

1. **Corpus scale is the highest-leverage input.** Every quality improvement
   below is secondary to feeding it more text.
2. **Output inherits the corpus's licence.** Because the model reproduces long
   spans verbatim, generated text from a copyrighted corpus is a derivative of
   it in the most literal sense. This is why the corpus stays untracked and why
   evaluation is restricted to public-domain sources.

## Standard-library evolution

No third-party runtime dependency is needed for any of the following.

### Tier 1 — prerequisites

Small changes that everything else depends on.

- **`argparse`** — the keystone. Replaces positional-only parsing, provides
  `--help` and type validation, and retires the `0`-means-default sentinel.
  Every feature below needs a flag to reach it.
- **`random.Random(seed)` + `--seed`** — instantiate a generator rather than
  using the global `random` module. Makes runs reproducible and lets the test
  suite drop its autouse global-seeding fixture, which is currently a smell.
- **`encoding="utf-8"` on both `open()` calls** — a live bug, not a feature.
  Both calls currently use the locale default encoding, so a corpus containing
  smart quotes or em-dashes fails on a differently-configured machine.

### Tier 2 — capability

- **`--order N` via `collections.deque(maxlen=N)`** — order is baked into the
  `(w1, w2)` tuple today. A bounded deque generalizes it in roughly four lines.
  Given the table above, this converts a fixed, badly-chosen constant into the
  parameter that most needs tuning per corpus.
- **`collections.Counter` + `random.choices(pop, weights=…)`** — `possibles`
  stores a list *with duplicates*, so `random.choice` is already
  frequency-weighted (correct), but a word following a prefix 50 times is stored
  50 times. A `Counter` yields an identical distribution at a fraction of the
  memory, which is precisely what makes a corpus large enough to fix the
  memorization problem feasible.
- **`fileinput` + `nargs="+"`** — multiple input files and stdin for free.
  `cat corpus/*.txt | markov --order 3` makes the tool composable.
- **Sentence-aware stopping** — output currently ends mid-clause. Stopping on a
  token ending in `.!?`, exposed as `--sentences N` alongside `--words N`, is a
  handful of lines and the largest perceived-quality gain available.

### Tier 3 — worthwhile, less urgent

- **`--stats`** — emit the branching/forced table for the user's own corpus.
  This is what turns the repository from a script into an instrument; see
  [Teaching](#teaching-strongest-fit).
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

### Text analysis — the model run backwards

The same n-gram table answers "how predictable is this document?" Branching
factor and perplexity are real signals for readability scoring, authorship
comparison, and boilerplate detection. Arguably a better fit for the data
structure than generation is.

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
2. `--order` and `--stats` together. They are the teaching story, and `--stats`
   makes the effect of `--order` legible.
3. Fix the two known defects, now that flags exist to control the alternatives.
4. `Counter` + `random.choices`, then multi-file input — the pair that makes a
   corpus large enough to matter practical.
5. Sentence-aware stopping and the entry point as polish.

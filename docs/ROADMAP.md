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
Reproduce this row with `markov stats --tokenizer whitespace`.

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
2. **Output inherits the corpus's license.** Because the model still reproduces
   multi-word spans verbatim, generated text from a copyrighted corpus is a
   derivative of it in the most literal sense. This is why the corpus stays
   untracked and why evaluation is restricted to public-domain sources.

## Standard-library evolution

No third-party runtime dependency is needed for any of the following.

### Tier 1 — prerequisites

Small changes that everything else depends on.

- ~~**`argparse`**~~ — **done** for `markov_cli.py`, which is now a subcommand
  dispatcher with `--help` and type validation. The positional-only interface
  and its `0`-means-default sentinel are gone, and `markov.py`, the last place
  that form survived, has been retired.
- ~~**`random.Random(seed)` + `--seed`**~~ — **done** for `generate`, which draws
  from its own instance rather than the global `random` module; a test pins that
  `--seed` reproduces a run whatever else has seeded `random`. The autouse
  global-seeding fixture was removed along with `generate_text()`, the last code
  that needed it.
- **`encoding="utf-8"` on every `open()` call** — a live bug, not a feature. A
  corpus containing smart quotes or em-dashes fails on a machine whose locale
  default is not UTF-8. Fixed in `markov_cli.py` on both the read and the write
  side, each pinned by a test that re-runs the CLI under
  `-X warn_default_encoding`, where an implicit `open()` is an error. No code
  in the repository reads or writes with the locale default any more.

### Tier 2 — capability

- ~~**`--order N`**~~ — **done**. Order is a parameter of `build_chain`,
  `NgramModel`, and all three subcommands, not the constant baked into a
  `(w1, w2)` tuple. `generate` defaults to 2 and `analyze`/`stats` to 3: the
  table above is why, since order 3 is forced at 94% of its states and generates
  little more than transcription.
- ~~**`collections.Counter` + `random.choices(pop, weights=…)`**~~ —
  **rejected on measurement**. The claim here was that `possibles`, a list
  storing a word once per occurrence, wastes memory a `Counter` would save at
  an identical distribution. The distribution part is right; the memory part is
  backwards at this corpus's shape.

  An empty `Counter` costs about 200 bytes against a 1-element list's 64–88, so
  a context has to repeat roughly 16–24 times before the dictionary pays for
  itself. The corpus averages **1.82 occurrences per context**, 80% of contexts
  are seen exactly once, and **0.43%** reach the break-even point. Measured over
  the whole table:

  | Order | List | `Counter` | |
  |---|---:|---:|---|
  | 2 | 13.5 MB | 23.1 MB | 1.71× worse |
  | 3 | 22.9 MB | 37.9 MB | 1.65× worse |

  Repeating the corpus eight times, which lifts the average context to 14.6
  occurrences, still leaves `Counter` 1.20× worse. Natural language is
  Zipfian: the many contexts are rare and the frequent ones are few, so the
  per-context overhead dominates whatever the totals are.

  The premise was also wrong about which structure is the constraint. The
  generation chain is not what a 10⁷-token corpus strains — the analysis
  `NgramModel` is, at roughly 115 bytes per stored n-gram, with distinct
  n-grams growing as `6.74 × tokens^0.867` (Heaps' law). That projects about
  7.9M n-grams, near 0.9 GB, at 10⁷ tokens. Smaller keys there — interning
  tokens to integer IDs — is the change that would earn its complexity, and it
  is not needed until corpora approach 10⁶.
- **stdin** — the multi-file half of this is done: `--reference` takes
  `nargs="+"` and accepts directories, globs, and file lists on every
  subcommand. Reading a corpus from a pipe is what is left, and would make
  `cat corpus/*.txt | markov generate` work.
- ~~**Sentence-aware stopping**~~ — **done**, as `generate --sentences N`
  alongside `--words N`. Stops on a token ending in `.!?` (closing quotes and
  brackets allowed after), and starts at a context that follows one in the
  source, so the output is whole sentences rather than two fragments around N
  complete ones. Naive about abbreviations: `Mr.` ends a sentence as far as this
  is concerned. Capped at 2,000 tokens so a corpus with no sentence punctuation
  ends the run and says so.

### Tier 3 — worthwhile, less urgent

- ~~**`--stats`**~~ — **done**, as the `stats` subcommand: it emits the
  branching/forced table for the user's own corpus, in either tokenizer, as text
  or JSON. This is what turns the repository from a script into an instrument;
  see [Teaching](#teaching-strongest-fit).
- ~~**Leave-one-out without rebuilding**~~ — **done**. Calibration built one
  model per reference document, so its cost was document count × corpus size.
  Counts are integers, so the same table is reached by subtracting a document's
  own n-grams from one whole-corpus model: `NgramModel.without` does that and
  puts it back afterwards, in a `finally` so a failure mid-run cannot leave the
  shared model damaged. Equal to a rebuild rather than close to it — the seam
  n-grams a document's removal destroys and creates are handled explicitly, and
  a test checks every count, the token total, and the vocabulary against a real
  rebuild. On the 54 inaugurals: **5.3s to 0.4s, and 175 MB to 75 MB peak**,
  with output byte-identical.

- ~~**Size-adjusted verdict threshold**~~ — **done**. `VARIANT_Z = 2.5` was
  applied flat, which assumes the expected range is known rather than estimated
  from as few as a handful of scores. The error runs both ways at once: a
  document inside the reference is bounded at (n-1)/sqrt(n), so 2.5 was
  unreachable below 9 documents, while an outside target is divided by a spread
  those same few scores get badly wrong — on 5-document subsets of the
  inaugurals, one unchanged speech ranged 0.7 to 7.9 z and 2.5 flagged it 21
  times in 60. `verdict_threshold` derives both from the reference size,
  holding the false-flag rate at the 1.24% that 2.5 implies: the studentized
  residual quantile for a member, bounded by the same ceiling, and the
  prediction-interval form t(n-1)·sqrt(1+1/n) for an outside target. Needed
  Student's t, which the standard library does not carry, so the regularized
  incomplete beta is computed by continued fraction and inverted by bisection —
  checked against a printed t table and, for the thresholds themselves, against
  simulated false-flag rates. On the 54 inaugurals **no verdict changes**
  (2.44 and 2.61 against 2.50); on 5-document subsets the arbitrary flagging
  drops from 21/60 to 5/60, and a member becomes flaggable at all.

- **Model persistence** — build once, generate many. Use `json` with encoded
  tuple keys, **not `pickle`**: unpickling executes arbitrary code, and a saved
  model is exactly the sort of file people pass around. Now the larger remaining
  win for the analysis path: subtraction removed the per-document rebuild, so
  what is left is the one build per run, which a general reference reused across
  many targets pays over and over.
- **`re` tokenization** — `r"\w+|[^\w\s]"` separates punctuation from words so
  `last!` and `last` stop being distinct tokens. Requires a detokenizer to
  reassemble output, so this is a genuine trade-off rather than a clear win.
- ~~**`[project.scripts]` entry point**~~ — **done**. `uv sync` installs a
  `markov` command, so `uv run markov stats …` replaces
  `uv run python markov_cli.py stats …`. It needed a build backend as well as the
  `[project.scripts]` line: uv installs entry points only for a project it
  builds, and the project had been marked `package = false`. Hatchling builds
  it, and the wheel carries only `markov_cli.py` and `markov_core.py`. The
  runtime stays standard-library-only; hatchling is needed only at install
  time.

### Known defects

None open. The original script had two, and both went with it when `markov.py`
and `generate_text()` were retired:

- **The corpus looped.** A sentinel made the walk emit empty strings at the end
  of the source and restart from the first word, silently splicing the corpus
  end onto its beginning. `generate` stops where the corpus stops and reports the
  short run on stderr.
- **All-lowercase input crashed.** Seeding required a capitalized prefix, and
  `random.choice` raised `IndexError` when there was none. `generate` falls back
  from sentence-opening contexts, to capitalized ones, to any context at all.

## Where this tool still earns its place

For "produce plausible text", a Markov chain has been obsolete since large
language models. Every use below is one where an LLM is the *wrong* tool rather
than a stronger one.

### Teaching (strongest fit)

The canonical first generative model: the walk itself is about twenty lines of
`markov_core.walk`, with no dependencies, no GPU, and nothing hidden. With
`generate --order` and `stats` side by side it is a laboratory in which a student
watches order 1 → 4 slide from gibberish into verbatim plagiarism *and reads the
numbers explaining why*. That is a lesson about memorization and overfitting that
transfers directly to reasoning about modern models.

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

Tier 1 is done, and `--order`, `--stats`, sentence-aware stopping and multi-file
input arrived with the CLI. `markov.py` has been retired, taking both known
defects with it, and `uv run markov` is the command. What is left, in order:

1. Model persistence, so a large reference is built once rather than per run.
   With leave-one-out no longer rebuilding per document, this is the remaining
   per-run cost, and the one a reused general reference pays most.
2. Reading a corpus from stdin, the half of the stdin item still outstanding.
3. Smaller n-gram keys — interning tokens to integer IDs — if and when corpora
   approach 10⁶ tokens. Not before: at 128,445 the model is 23 MB and the
   complexity buys nothing.

`Counter` + `random.choices` was item 1 here until it was measured; it is
rejected above, with the numbers.

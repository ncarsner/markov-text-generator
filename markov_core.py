"""N-gram engine: tokenization, counts, backoff scoring, and sampling.

This is the shared core beneath every CLI subcommand. Two views of the same
counts serve the two applications:

- ``NgramModel`` scores how surprising a document is under a reference corpus,
  which is the principal application (see docs/PRD-deviation-analysis.md).
- ``build_chain`` and ``sample`` walk the same n-grams to emit text, which is
  the side effect.

Standard library only, and deterministic apart from ``sample``.
"""

import bisect
import collections
import math
import random
import re

DEFAULT_ORDER = 3

# Stupid backoff (Brants et al. 2007). Unnormalized, so scores are not
# probabilities in the strict sense, but they are cheap, stable, and -- the
# reason they are used here -- trivial to explain in an audit.
BACKOFF_ALPHA = 0.4


# --- Tokenization ---------------------------------------------------------
# Pluggable seam. Domain profiles (legal, and others) register here later
# without the engine changing; see the PRD's Later Enhancements.

def whitespace_tokens(text):
    """Split on whitespace, preserving case and punctuation.

    The generation view: output should read like the source, so ``Freedom!``
    and ``freedom`` stay distinct.
    """
    return text.split()


_WORD = re.compile(r"[a-z0-9']+")


def plain_tokens(text):
    """Lowercased word tokens with punctuation dropped.

    The analysis view: ``Freedom!`` and ``freedom`` are the same evidence about
    what language a corpus expects.
    """
    return _WORD.findall(text.lower())


TOKENIZERS = {"whitespace": whitespace_tokens, "plain": plain_tokens}

# Each tokenizer paired with a pattern that finds the same tokens in the
# original text. Span reporting needs to point back at the source -- a line
# number and the words as written -- and a token list alone has lost that.
# `plain` matches both cases here and lowercases the result rather than
# lowercasing the text first, which would not preserve offsets for every
# character.
LOCATORS = {
    "whitespace": (re.compile(r"\S+"), False),
    "plain": (re.compile(r"[A-Za-z0-9']+"), True),
}


def locate_tokens(text, name):
    """Tokenize ``text``, returning (token, start, end) for each token.

    Produces exactly the tokens ``TOKENIZERS[name]`` produces, with the offsets
    into ``text`` that the tokenizer itself discards.
    """
    try:
        pattern, lowercase = LOCATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown tokenizer {name!r}; choose from {sorted(LOCATORS)}"
        ) from None
    return [
        (match.group().lower() if lowercase else match.group(),
         match.start(), match.end())
        for match in pattern.finditer(text)
    ]


def line_starts(text):
    """Offsets at which each line begins, for turning an offset into a line."""
    starts = [0]
    for i, char in enumerate(text):
        if char == "\n":
            starts.append(i + 1)
    return starts


def line_of(starts, offset):
    """The 1-based line number containing ``offset``."""
    return bisect.bisect_right(starts, offset)


def get_tokenizer(name):
    """Look up a tokenizer by name, raising ValueError for an unknown profile."""
    try:
        return TOKENIZERS[name]
    except KeyError:
        raise ValueError(
            f"unknown tokenizer {name!r}; choose from {sorted(TOKENIZERS)}"
        ) from None


# --- Chains (generation) --------------------------------------------------

def build_chain(tokens, order=2):
    """Map each order-length context to the tokens that followed it.

    Values are lists that keep duplicates, so ``random.choice`` over them is
    frequency-weighted without needing explicit weights.
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    chain = collections.defaultdict(list)
    for i in range(len(tokens) - order):
        chain[tuple(tokens[i:i + order])].append(tokens[i + order])
    return chain


def capitalized_contexts(chain):
    """Contexts whose first token starts with a capital -- plausible openings."""
    return [context for context in chain if context[0][:1].isupper()]


def walk(chain, start, order, rng=random, limit=None, stop=None):
    """Yield tokens following ``start``, ending when the chain runs out.

    The generator form of ``sample``, and the difference is the ending. A
    corpus has a last word, and the context that follows it has no recorded
    continuation; this stops there. ``sample`` raises instead, which is the
    older contract and is kept because ``markov.py`` depends on it.

    ``limit`` caps how many tokens are produced. ``stop`` is called with each
    token as it is emitted and ends the walk when it returns true, which is how
    sentence-aware stopping is expressed without the engine knowing what a
    sentence is.

    A ``stop`` condition the chain may never satisfy needs a ``limit`` to fall
    back on, so the two are required together: a cyclic chain and a condition
    that never fires is an unbounded walk, and failing at the call is better
    than hanging inside it.
    """
    if stop is not None and limit is None:
        raise ValueError("walk with a stop condition needs a limit to fall back on")

    out = list(start)
    context = tuple(start)
    produced = 0
    while limit is None or produced < limit:
        choices = chain.get(context)
        if not choices:
            return
        word = rng.choice(choices)
        yield word
        produced += 1
        out.append(word)
        context = tuple(out[-order:])
        if stop is not None and stop(word):
            return


def sample(chain, start, count, order, rng=random):
    """Walk the chain from ``start``, returning it followed by ``count`` tokens.

    Raises IndexError if the walk reaches a context with no recorded
    continuations.
    """
    out = list(start)
    out.extend(walk(chain, start, order, rng=rng, limit=count))
    if len(out) - len(start) < count:
        raise IndexError(
            f"chain reached a context with no continuations after "
            f"{len(out) - len(start)} of {count} tokens"
        )
    return out


# --- Counts and scoring (analysis) ----------------------------------------

class NgramModel:
    """N-gram counts with stupid backoff scoring.

    ``surprisal`` answers the question the tool exists to ask: how unexpected
    is this token, given the reference corpus this model was built from?
    """

    def __init__(self, tokens, order=DEFAULT_ORDER, alpha=BACKOFF_ALPHA):
        if order < 1:
            raise ValueError("order must be at least 1")
        tokens = list(tokens)
        self.order = order
        self.alpha = alpha
        # counts[n] holds n-gram counts; counts[0] is unused padding so that
        # counts[n] can be indexed by n directly.
        self.counts = [collections.Counter() for _ in range(order + 1)]
        for n in range(1, order + 1):
            counter = self.counts[n]
            for i in range(len(tokens) - n + 1):
                counter[tuple(tokens[i:i + n])] += 1
        self.total = len(tokens)
        self.vocab = len(self.counts[1])

    # -- scoring --

    def _truncate(self, context):
        """Keep only as much context as the model's order can use."""
        context = tuple(context)
        return context[-(self.order - 1):] if self.order > 1 else ()

    def _match(self, context, word):
        """Longest context that ``word`` was actually seen following.

        Returns (length used, that context, its count with ``word``), or
        (-1, (), 0) when the word never appears in the reference at all. This
        is the single fact behind both the score and its explanation: backing
        off is choosing which n-gram to believe, and the answer is a specific
        count in the reference.
        """
        for n in range(len(context), -1, -1):
            prefix = context[len(context) - n:] if n else ()
            count = self.counts[n + 1].get(prefix + (word,), 0)
            if count:
                return n, prefix, count
        return -1, (), 0

    def score(self, context, word):
        """Return (score, backoff_level, count) for ``word`` after ``context``.

        ``backoff_level`` is how many context tokens were actually used; -1
        means nothing matched and the out-of-vocabulary floor applied. It is
        carried alongside the score so a flagged span can be explained.
        """
        context = self._truncate(context)
        level, prefix, count = self._match(context, word)
        if level < 0:
            floor = (self.alpha ** (len(context) + 1)) / (self.total + self.vocab)
            return floor, -1, 0
        denominator = self.counts[level][prefix] if level else self.total
        return (
            (self.alpha ** (len(context) - level)) * count / denominator,
            level,
            count,
        )

    def explain(self, context, word):
        """Say why ``word`` scored as it did, in terms a reviewer can check.

        Every field is a fact about the reference corpus: which n-gram was
        believed, how often it occurs, and how often the word occurs at all.
        This is what makes a flag traceable rather than merely reported.
        """
        context = self._truncate(context)
        level, prefix, count = self._match(context, word)
        return {
            "word": word,
            "backoff_level": level,
            "context_offered": list(context),
            "context_used": list(prefix),
            "count": count,
            "word_count": self.counts[1].get((word,), 0),
        }

    def probability(self, context, word):
        """The backoff score alone."""
        return self.score(context, word)[0]

    def surprisal(self, context, word):
        """Bits of surprise: -log2 of the score. Higher means less expected."""
        return -math.log2(self.probability(context, word))

    # -- document-level --

    def surprisals(self, tokens):
        """Per-token surprisal, aligned with ``tokens``."""
        tokens = list(tokens)
        window = self.order - 1
        return [
            self.surprisal(tokens[max(0, i - window):i], word)
            for i, word in enumerate(tokens)
        ]

    def bits_per_token(self, tokens):
        """Mean surprisal over a document. The headline deviation metric."""
        scores = self.surprisals(tokens)
        if not scores:
            raise ValueError("cannot score an empty document")
        return sum(scores) / len(scores)

    def perplexity(self, tokens):
        """2 ** bits_per_token -- the effective number of choices per token."""
        return 2 ** self.bits_per_token(tokens)

    def novel_ngram_rate(self, tokens, n=None):
        """Fraction of a document's n-grams absent from the reference."""
        n = self.order if n is None else n
        if not 1 <= n <= self.order:
            raise ValueError(f"n must be between 1 and {self.order}")
        tokens = list(tokens)
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not grams:
            raise ValueError("document is shorter than the n-gram size")
        return sum(g not in self.counts[n] for g in grams) / len(grams)


# --- Corpus diagnostics ---------------------------------------------------

def chain_statistics(tokens, order):
    """Return (mean distinct continuations, fraction of states with only one).

    A corpus whose states are mostly "forced" -- one continuation and no choice
    -- can only transcribe itself, which makes it a poor reference and a poor
    generator alike.
    """
    table = collections.defaultdict(set)
    tokens = list(tokens)
    for i in range(len(tokens) - order):
        table[tuple(tokens[i:i + order])].add(tokens[i + order])
    if not table:
        raise ValueError("corpus is shorter than the chain order")
    sizes = [len(v) for v in table.values()]
    return sum(sizes) / len(sizes), sum(s == 1 for s in sizes) / len(sizes)

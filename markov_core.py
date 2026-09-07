"""N-gram engine: tokenization, counts, backoff scoring, and sampling.

This is the shared core beneath every CLI subcommand. Two views of the same
counts serve the two applications:

- ``NgramModel`` scores how surprising a document is under a reference corpus,
  which is the principal application (see docs/PRD-deviation-analysis.md).
- ``build_chain`` and ``sample`` walk the same n-grams to emit text, which is
  the side effect.

Standard library only, and deterministic apart from ``sample``.
"""

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


def sample(chain, start, count, order, rng=random):
    """Walk the chain from ``start``, returning it followed by ``count`` tokens.

    Raises IndexError if the walk reaches a context with no recorded
    continuations.
    """
    out = list(start)
    context = tuple(start)
    for _ in range(count):
        out.append(rng.choice(chain[context]))
        context = tuple(out[-order:])
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

    def score(self, context, word):
        """Return (score, backoff_level, count) for ``word`` after ``context``.

        ``backoff_level`` is how many context tokens were actually used; -1
        means nothing matched and the out-of-vocabulary floor applied. It is
        carried alongside the score so a flagged span can be explained.
        """
        context = self._truncate(context)
        for n in range(len(context), -1, -1):
            prefix = context[len(context) - n:] if n else ()
            count = self.counts[n + 1].get(prefix + (word,), 0)
            if count:
                denominator = self.counts[n][prefix] if n else self.total
                return (self.alpha ** (len(context) - n)) * count / denominator, n, count
        floor = (self.alpha ** (len(context) + 1)) / (self.total + self.vocab)
        return floor, -1, 0

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

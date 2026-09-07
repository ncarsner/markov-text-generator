import random
import sys
import textwrap

import markov_core

ORDER = 2


def generate_text(input_text, num_words=None):
    """Generate text from an order-2 chain over ``input_text`` (a list of lines).

    A thin wrapper over markov_core that preserves this function's original
    behavior exactly, quirks included: the walk is seeded from a capitalized
    context, and the two sentinel entries below make it loop back to the start
    of the corpus rather than stop at the end. Both are pinned by tests.
    """
    words = [word for line in input_text for word in line.split()]

    # Padding with two empty tokens reproduces the original's starting state,
    # in which the first words of the corpus follow an empty context.
    padded = ["", ""] + words
    chain = markov_core.build_chain(padded, ORDER)

    # Close the chain at the end of the corpus. These are what make the walk
    # wrap around to the first word instead of terminating.
    chain[padded[-2], padded[-1]].append("")
    chain[padded[-1], ""].append("")

    if num_words is None:
        num_words = len(words)

    start = random.choice(markov_core.capitalized_contexts(chain))
    return " ".join(markov_core.sample(chain, start, num_words, ORDER))


def main(argv=None):
    # Check for command-line arguments (argv is injectable for testing)
    args = sys.argv[1:] if argv is None else list(argv)

    if len(args) < 2:
        print("Usage: python markov_cli.py <input_file> <num_words or 0> [output_file]")
        sys.exit(1)

    input_file = args[0]
    num_words = int(args[1])

    # Read input text from source file
    with open(input_file, "r") as f:
        input_text = f.readlines()

    # Use default value if num_words is zero
    if num_words == 0:
        num_words = None

    # Generate random text
    generated_text = generate_text(input_text, num_words)

    # Print output wrapped to 70 columns
    print(textwrap.fill(generated_text))

    # Export to output file if specified
    if len(args) >= 3:
        output_file = args[2]
        with open(output_file, "w") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()

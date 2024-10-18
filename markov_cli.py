import collections
import random
import sys
import textwrap


def generate_text(input_text, num_words=None):
    # Build possibles table indexed by pair of prefix words (w1, w2)
    w1 = w2 = ""
    possibles = collections.defaultdict(list)
    for line in input_text:
        for word in line.split():
            possibles[w1, w2].append(word)
            w1, w2 = w2, word

    # Avoid empty possibles lists at end of input
    possibles[w1, w2].append("")
    possibles[w2, ""].append("")

    # Set default length to the number of words in the input_text
    if num_words is None:
        num_words = sum(len(line.split()) for line in input_text)

    # Generate randomized output (start with a random capitalized prefix)
    w1, w2 = random.choice([k for k in possibles if k[0][:1].isupper()])
    output = [w1, w2]
    for _ in range(num_words):
        word = random.choice(possibles[w1, w2])
        output.append(word)
        w1, w2 = w2, word

    return " ".join(output)


def main():
    # Check for command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python markov_cli.py <input_file> <num_words or 0> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    num_words = int(sys.argv[2])

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
    if len(sys.argv) >= 4:
        output_file = sys.argv[3]
        with open(output_file, "w") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()

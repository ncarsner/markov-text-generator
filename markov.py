import collections
import random
import sys
import textwrap

# Configuration variables
SOURCE_FILE = './data/input/speech_we_choose_to_go_to_the_moon.txt'  # Default source file to read from
NUM_WORDS = 100  # Default number of words to generate
OUTPUT_FILE = None  # Optional output file (None = print to console only)

def main():
    # Use command-line arguments if provided, otherwise use configuration variables
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = SOURCE_FILE
    
    if len(sys.argv) >= 3:
        num_words = int(sys.argv[2])
    else:
        num_words = NUM_WORDS
    
    if len(sys.argv) >= 4:
        output_file = sys.argv[3]
    else:
        output_file = OUTPUT_FILE

    # Read input text from file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Build possibles table indexed by pair of prefix words (w1, w2)
    w1 = w2 = ''
    possibles = collections.defaultdict(list)
    for line in lines:
        for word in line.split():
            possibles[w1, w2].append(word)
            w1, w2 = w2, word

    # Avoid empty possibles lists at end of input
    possibles[w1, w2].append('')
    possibles[w2, ''].append('')

    # Generate randomized output (start with a random capitalized prefix)
    w1, w2 = random.choice([k for k in possibles if k[0][:1].isupper()])
    output = [w1, w2]
    for _ in range(num_words):
        word = random.choice(possibles[w1, w2])
        output.append(word)
        w1, w2 = w2, word

    # Print output wrapped to 70 columns
    print(textwrap.fill(' '.join(output)))

if __name__ == '__main__':
    main()
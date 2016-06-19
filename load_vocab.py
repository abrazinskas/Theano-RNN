import essen
import argparse

# Parse args.
parser = argparse.ArgumentParser()
parser.add_argument("vocab_filename")
args = parser.parse_args()

# # Parse the csv file
idx_to_notes, notes_to_idx = essen.load_notes_vocab(args.vocab_filename)
print(idx_to_notes)
print(notes_to_idx)

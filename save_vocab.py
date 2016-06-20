import essen
import argparse

# Parse args.
parser = argparse.ArgumentParser()
parser.add_argument("csv_file")
parser.add_argument("vocab_filename")
args = parser.parse_args()

# Parse the data and save the vocabulary file.
idx_to_notes, notes_to_idx = essen.create_vocab(args.csv_file)
essen.save_notes_vocab(idx_to_notes, args.vocab_filename)

import essen
import argparse

# Parse args.
parser = argparse.ArgumentParser()
parser.add_argument("vocab_filename")
parser.add_argument("songs_filename")
args = parser.parse_args()

# # Parse the csv file
idx_to_notes, notes_to_idx = essen.load_notes_vocab(args.vocab_filename)
for song in essen.parse_songs(args.songs_filename, notes_to_idx):
    print(song)

import argparse
import csv
from fractions import Fraction

# Row constants
ROW_SONG_ID    = 1
ROW_PITCH      = 2
ROW_DURATION   = 3

# Data constants
PITCH          = 0
DURATION       = 1

# Get the pitch range from the dataset
def pitch_range_from_csv(csv_file):
    pmin, pmax = None, None
    with open(csv_file, 'rb') as csv_file:
        skip_first = True
        reader = csv.reader(csv_file)
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            pitch = int(row[ROW_PITCH])
            if pmin is None:
                pmin = pitch
                pmax = pitch
            else:
                pmax = max(pmax, pitch)
                pmin = min(pmin, pitch)
    return pmin, pmax

# Convert a CSV row to a data point
def row_to_data_point(row):
    pitch = int(row[ROW_PITCH])
    duration = str(Fraction(row[ROW_DURATION]).limit_denominator(32))
    return (pitch, duration)

def parse_songs(csv_file, notes_to_idx):
    with open(csv_file, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        song_id = None
        skip_first = True
        song = []
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            if song_id == row[ROW_SONG_ID] or song_id is None:
                song_id = row[ROW_SONG_ID] if song_id is None else song_id
                data_point = row_to_data_point(row)
                song.append(notes_to_idx[data_point])
            else:
                cur_song = song
                song_id = row[ROW_SONG_ID]
                song = []
                yield cur_song

# Create a vocabulary from a given dataset.
def create_vocab(csv_file):
    idx = 0
    notes_to_idx = dict()
    with open(csv_file, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        song_id = None
        skip_first = True
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            data_point = row_to_data_point(row)
            if data_point not in notes_to_idx:
                notes_to_idx[data_point] = idx
                idx += 1
    idx_to_notes = {v: k for k, v in notes_to_idx.iteritems()}
    return idx_to_notes, notes_to_idx

# Saves the notes vocabulary (idx_to_notes) to a csv file with the given filename.
def save_notes_vocab(idx_to_notes, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for k, v in idx_to_notes.items():
            writer.writerow([k, v])

# Load the notes vocabulary from a csv file. Returns idx_to_nodes and nodes_to_idx.
def load_notes_vocab(filename):
    idx_to_notes = dict()
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            idx_to_notes[int(row[0])] = eval(row[1])
        notes_to_idx = {v: k for k, v in idx_to_notes.iteritems()}
        return idx_to_notes, notes_to_idx

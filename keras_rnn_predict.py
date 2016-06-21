import argparse
import essen
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

parser = argparse.ArgumentParser()
parser.add_argument("vocab_filename")
parser.add_argument("weights_file")
args = parser.parse_args()

maxlen = 3
idx_to_notes, notes_to_idx = essen.load_notes_vocab(args.vocab_filename)
stop_note = len(idx_to_notes)
vocab_size = len(idx_to_notes) + 1
idx_to_notes[stop_note] = (-1, "STOP_NOTE")

model = Sequential()
model.add(LSTM(200, return_sequences=False, input_shape=(maxlen, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.load_weights(args.weights_file)

starting_notes = [notes_to_idx[(72, "1")], notes_to_idx[(60, "1/3")], notes_to_idx[(73, "1/3")]]
x = np.zeros((1, maxlen, vocab_size), dtype=np.bool)
sys.stdout.write("Starting notes: ")
for i, note in enumerate(starting_notes):
    x[0][i][note] = True
    sys.stdout.write(str(idx_to_notes[note]) + " ")
# sys.stdout.write("\n")

# print("Starting sequence: " + str(idx_to_notes[0]) + ", " + str(idx_to_notes[1]) + ", " + str(idx_to_notes[2]))

pred_note_idx = None
cur_notes = starting_notes
iterations = 0
while pred_note_idx != stop_note:
    pred = model.predict(x, batch_size=1, verbose=0)[0]
    distr = stats.rv_discrete(values=(np.arange(0, vocab_size), pred))
    pred_note_idx = distr.rvs(size=1)[0]
    pred_note = idx_to_notes[pred_note_idx]
    sys.stdout.write(str(pred_note) + " ")
    for i, note in enumerate(cur_notes):
        x[0][i][note] = False
        if i > 0:
            x[0][i-1][note] = True
            cur_notes[i-1] = note
    x[0][i][pred_note_idx] = True
    cur_notes[len(cur_notes) - 1] = pred_note_idx
    iterations += 1
    if iterations >= 100:
        sys.stdout.write("Stopping song...")
        break
sys.stdout.write("\n")

# plt.plot(range(0, vocab_size), pred)
# plt.xlim(0, vocab_size)
# plt.show()

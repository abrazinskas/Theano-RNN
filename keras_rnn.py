from __future__ import print_function
import numpy as np
import random
import sys
import essen
import argparse
import theano
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.utils.data_utils import get_file

parser = argparse.ArgumentParser()
parser.add_argument("vocab_filename")
parser.add_argument("train_songs_file")
# parser.add_argument("test_songs_file")
parser.add_argument("-w", "--weights_file", default=None, nargs="?")
args = parser.parse_args()

idx_to_notes, notes_to_idx = essen.load_notes_vocab(args.vocab_filename)
stop_note = len(idx_to_notes)
vocab_size = len(idx_to_notes) + 1
music = []
for song in essen.parse_songs(args.train_songs_file, notes_to_idx):
    music.extend(song)
    music.append(stop_note)
print('music length:', len(music))
print('total notes:', vocab_size)

# music_test = []
# for song in essen.parse_songs(args.test_songs_file, notes_to_idx):
#     music_test.extend(song)
#     music_test.append(stop_note)

maxlen = 3
step = 1
melodies = []
next_notes = []
for i in range(0, len(music) - maxlen, step):
    melody = music[i: i + maxlen]
    melodies.append(melody)
    next_notes.append(music[i + maxlen])
print('num sequences:', len(melodies))

# melodies_test = []
# next_notes_test = []
# for i in range(0, len(music_test) - maxlen, step):
#     melody = music_test[i: i + maxlen]
#     melodies_test.append(melody)
#     next_notes_test.append(music_test[i + maxlen])

print('Vectorization...')
X = np.zeros((len(melodies), maxlen, vocab_size), dtype=np.bool)
y = np.zeros((len(melodies), vocab_size), dtype=np.bool)
for i, melody in enumerate(melodies):
    for t, note in enumerate(melody):
        X[i, t, note] = 1
    y[i, next_notes[i]] = 1

# X_test = np.zeros((len(melodies_test), maxlen, vocab_size), dtype=np.bool)
# y_test = np.zeros((len(melodies_test), vocab_size), dtype=np.bool)
# for i, melody in enumerate(melodies_test):
#     for t, note in enumerate(melody):
#         X_test[i, t, note] = 1
#     y_test[i, next_notes[i]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(300, return_sequences=True, input_shape=(maxlen, vocab_size)))
model.add(Dropout(0.5))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.5))
model.add(LSTM(300, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# num_epochs = 100
# for i in range(1, 1 + num_epochs):
    # print("=" * 10)
    # print("Epoch: " + str(i) + "/" + str(num_epochs))
history = model.fit(X, y, batch_size=100, nb_epoch=1, verbose=1, validation_split=0.2, shuffle=True)
with open('output.log', 'w') as f:
    for k in history.history.keys():
        f.write(k)
        f.write(": ")
        f.write(str(history.history[k]))
        f.write("\n")
# test_loss = model.evaluate(X_test, y_test, batch_size=128)
# print()
# print("Test loss: " + str(test_loss))

if args.weights_file is not None:
    print("Saving weights to " + args.weights_file + "...")
    model.save_weights(args.weights_file)

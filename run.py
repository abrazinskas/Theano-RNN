import nltk
import itertools
import csv
import numpy as np
from support import construct_vocabulary,write_vocabulary,read_vocabulary, tokenize_file, generate_sentence
import os
import copy
from RNN import RNN
from rnn_theano import RNNTheano
import matplotlib.pyplot as plt



unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
THEANO = True

train_file = "dataset/small.csv"
vocab_file = "dataset/vocab.txt"
weights_file = "weights/rnn_theono.weights"

SPEC_SYMBOLS_COUNT = 3
VOCAB_SIZE = 8000
MAX_SENTENCES = 100
MAX_L_SENTENCES = 100 # maximum number of sentences that should be used for loss computation
ALPHA = 0.1
EPOCHS = 5

# creating vocabulary
if not os.path.isfile(vocab_file):
    vocab = construct_vocabulary(train_file)
    write_vocabulary(vocab,vocab_file)

# read the vocab
index_to_word, word_to_index = read_vocabulary(vocab_file, 8000)
# adding special symbols
index_to_word.append(sentence_end_token)
index_to_word.append(sentence_start_token)
word_to_index[sentence_start_token] = VOCAB_SIZE+1
word_to_index[sentence_end_token] = VOCAB_SIZE+2

if THEANO:
    rnn = RNNTheano(VOCAB_SIZE+SPEC_SYMBOLS_COUNT, hidden_dim = 50)
else:
    rnn = RNN(VOCAB_SIZE+SPEC_SYMBOLS_COUNT, VOCAB_SIZE+SPEC_SYMBOLS_COUNT,hidden_dim = 100)
# generate sentences
print("training the model")
loss = [rnn.total_loss(itertools.islice(tokenize_file(word_to_index, train_file), MAX_L_SENTENCES))]
for e in range(EPOCHS):
    i = 0
    print("--- Epoch "+str(e+1)+" ---")
    loss.append(rnn.total_loss(itertools.islice(tokenize_file(word_to_index, train_file), MAX_L_SENTENCES)))
    sentences = tokenize_file(word_to_index, train_file)
    for sentence in itertools.islice(sentences, MAX_SENTENCES):
        i+=1
        sentence.insert(0,word_to_index[sentence_start_token])
        y = copy.copy(sentence)
        y.pop(0)
        y.append(word_to_index[sentence_end_token])
        rnn.train(sentence, y, ALPHA)
        if i % 10 == 0:
            print("preprocessed "+str(i))

# saving weights
rnn.save_weights(weights_file)

num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(rnn, word_to_index, index_to_word, sentence_start_token, sentence_end_token)
    print " ".join(sent)
loss.append(rnn.total_loss(itertools.islice(tokenize_file(word_to_index, train_file), MAX_L_SENTENCES)))
print(loss)

plt.plot(range(len(loss)),loss)
plt.ylabel('Error', fontsize=18)
plt.show()

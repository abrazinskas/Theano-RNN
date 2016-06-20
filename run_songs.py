import nltk
import itertools
import csv
import numpy as np
from support import construct_vocabulary,write_vocabulary,read_vocabulary, tokenize_file, generate_sentence, load_songs,load_vocab, file_len
import os
import copy
from RNN import RNN
from rnn_theano import RNNTheano
import matplotlib.pyplot as plt

sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

train_file = "dataset/simpleDupleTRAIN.csv"
test_file = "dataset/simpleDupleTEST.csv"
vocab_file = "dataset/vocab_songs.csv"
weights_file = "weights/rnn_theono.weights"

SPEC_SYMBOLS_COUNT = 2 # start symbol, end symbol
MAX_L_SENTENCES = file_len(test_file) # maximum number of sentences that should be used for loss computation
MAX_SENTENCES = file_len(train_file)
ALPHA = 0.015
EPOCHS = 5
HIDDEN_LAYER_SIZE = 50
PRELOAD_WEIGHTS = False

# read the vocab
index_to_word, word_to_index = load_vocab(vocab_file)
VOCAB_SIZE = len(index_to_word)
# adding special symbols
index_to_word[VOCAB_SIZE] = sentence_start_token
index_to_word[VOCAB_SIZE+1] = sentence_end_token
word_to_index[sentence_start_token] = VOCAB_SIZE
word_to_index[sentence_end_token] = VOCAB_SIZE+1


rnn = RNNTheano(VOCAB_SIZE+SPEC_SYMBOLS_COUNT, hidden_dim = HIDDEN_LAYER_SIZE)
if PRELOAD_WEIGHTS:
    print "preloading weights"
    rnn.preload_weights(weights_file)
    train_loss = []
    test_loss = []
else:
    print "training the model"
    train_loss = []
    test_loss = []
    for e in range(EPOCHS):
        i = 0
        print("--- Epoch "+str(e+1)+" ---")
        train_loss.append(rnn.total_loss(itertools.islice(load_songs(train_file,word_to_index),MAX_L_SENTENCES)))
        test_loss.append(rnn.total_loss(itertools.islice(load_songs(test_file,word_to_index),MAX_L_SENTENCES)))
        sentences = load_songs(train_file,word_to_index)
        for sentence in itertools.islice(sentences, MAX_SENTENCES):
            i+=1
            sentence.insert(0,word_to_index[sentence_start_token])
            y = copy.copy(sentence)
            y.pop(0)
            y.append(word_to_index[sentence_end_token])
            rnn.train(sentence, y, ALPHA)
            if i % 10 == 0:
                print("processed "+str(i))

# saving weights
rnn.save_weights(weights_file)


# num_sentences = 10
# senten_min_length = 7
#
# for i in range(num_sentences):
#     sent = []
#     # We want long sentences, not sentences with one or two words
#     while len(sent) < senten_min_length:
#         sent = generate_sentence(rnn, word_to_index, index_to_word, sentence_start_token, sentence_end_token)
#     print " ".join(sent)
# loss.append(rnn.total_loss(itertools.islice(tokenize_file(word_to_index, train_file), MAX_L_SENTENCES)))

print ("train and test losses: ")
print(train_loss)
print(test_loss)

fig = plt.figure()
plt.plot(range(len(train_loss)),train_loss,"-")
fig.suptitle('Train loss', fontsize=20)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.yticks(np.arange(0, max(train_loss)+2, 1.0))
fig.savefig("plots/train_"+str(EPOCHS)+"_epochs_"+str(HIDDEN_LAYER_SIZE)+"_hidden.png")


fig = plt.figure()
plt.plot(range(len(test_loss)),test_loss,"-")
fig.suptitle('Test loss', fontsize=20)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.yticks(np.arange(0, max(test_loss)+2, 1.0))
fig.savefig("plots/test_"+str(EPOCHS)+"_epochs_"+str(HIDDEN_LAYER_SIZE)+"_hidden.png")

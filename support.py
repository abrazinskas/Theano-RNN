import numpy as np
import nltk
import itertools
import warnings
import essen

IGNORED_TOKEN = "IGNORED_TOKEN"

def softmax(y):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            z = np.exp(y)
        except Warning:
            print("error")
            return np.zeros((len(y),))+0.000001
    Z = sum(z)
    if(Z==0): return 0
    return z/Z


def construct_vocabulary(file):
    freqs = nltk.FreqDist()
    print("Creating vocabulary...")
    with open(file) as f:
        print("- Processing file " + file + "...")
        words = nltk.word_tokenize(f.read().decode('utf-8').lower())
        words = [word for word in words if word.isalnum()]
        freqs.update(words)
    return freqs


# Tokenizes the files in the given folder
# - Converts to lower case, removes punctuation
# - Yields sentences split up in words, represented by vocabulary index
# Files in data folder should be tokenized by sentence (one sentence per newline),
# Like in the 1B-words benchmark
def tokenize_file(vocab_dict, file, subsample_frequent=True):
        with open(file) as f:
            for sentence in f:
                # Use nltk tokenizer to split the sentence into words
                words = nltk.word_tokenize(sentence.lower())
                # Filter punctuation
                words = [word for word in words if word.isalnum()]
                # Replace words that are not in vocabulary with IGNORED token
                words = [word if word in vocab_dict else IGNORED_TOKEN for word in words]
                if words:
                  yield [vocab_dict[word] for word in words]

# Sub-sampling of frequent words: can improve both accuracy and speed for large data sets
# Source: "Distributed Representations of Words and Phrases and their Compositionality"
# def allow_with_prob(word, vocab_dict, total_wordcount):
#     freq = float(vocab_dict[word]) / total_wordcount
#     removal_prob = 1.0 - np.sqrt(SUBSAMPLING_THRESHOLD / freq)
#     return np.random.random_sample() > removal_prob
#
# def allow_word(word, vocab_dict, total_wordcount, subsample_frequent):
#     allow = word.isalnum()
#     if not allow:
#         return False
#
#     if not word in vocab_dict:
#         return True # Will be replaced with IGNORED_TOKEN
#
#     if subsample_frequent:
#         allow = allow_with_prob(word, vocab_dict, total_wordcount)
#
#     return allow


def write_vocabulary(vocab, output_file, sep=' '):
    with open(output_file, 'w') as f:
        f.write("\n".join([sep.join((word[0],str(word[1]))) for word in vocab.most_common()]))
    print("Vocabulary written to " + output_file)


def read_vocabulary(filename, maxsize, sep=' '):
    index_to_word = []
    with open(filename) as f:
        for word in itertools.islice(f, 0, maxsize):
            splt=word.split(sep)
            index_to_word.append(splt[0])
    index_to_word.append(IGNORED_TOKEN)
    word_to_index = dict([ (w,i) for i, w in enumerate(index_to_word)])
    return index_to_word, word_to_index

def generate_sentence(model, word_to_index, index_to_word, start_symbols, sentence_end_token):
    # We start the sentence with the start token
    new_sentence = [word_to_index[start_symbols]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[IGNORED_TOKEN]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[IGNORED_TOKEN]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


def load_vocab(file):
    idx_to_notes, notes_to_idx = essen.load_notes_vocab(file)
    return idx_to_notes, notes_to_idx

def load_songs(file, vocab):
    for song in essen.parse_songs(file, vocab):
        yield song

def file_len(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
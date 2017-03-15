import itertools

import numpy as np
import pickle
from preproc.load_data import load_cornell, load_cornell_from_file, load_simpsons_from_file,load_simpsons
import os
import pickle
import re

import nltk
import numpy as np
import random

from preproc.load_data import load_cornell, load_cornell_from_file, load_simpsons_from_file,load_simpsons

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz .?!' # space is included in whitelist

limit = {
    'maxq': 20,
    'minq': 1,
    'maxa': 20,
    'mina': 1
}

UNK = 'unk'
VOCAB_SIZE = 10000

path_to_variables = "./variables/"

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

'''
Given a sentence, returns a list of tokens (each word in a token).
'''
def basic_tokenizer(sentence):
  words = []
  if isinstance(sentence, float):   # In case the sentence is just a number.
      sentence = str(sentence)
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

'''
    Given a set of sentences, it converts them all to lowercase, deletes invalid characters and tokenize.
    Returns the set after this operations has been performed.
'''
def basic_preproc(data):
    # All sentences to lower case
    data = [str(line).lower() for line in data]
    # Eliminate characters that are not in the white list.
    data = [ filter_line(line) for line in data ]
    data_tok = [basic_tokenizer(line) for line in data]
    return data_tok

'''
Given a sentence, it deletes all the characters that are not in the WHITELIST.
'''
def filter_line(sentence):
    return ''.join([ ch for ch in sentence if ch in WHITELIST ])

'''
Given a set of questions and answers, it deletes all those pairs where at least one of the sentences does not
fulfill the length requeriments.
'''
def filter_data(questions, answers):
    q_filt = []
    a_filt = []
    for i in range(0, len(questions)):
        length1 = len(questions[i])
        length2 = len(answers[i])
        if length1 >= limit['minq'] and length1 <= limit['maxq']:
            if length2 >= limit['mina'] and length2 <= limit['maxa']:
                q_filt.append(questions[i])
                a_filt.append(answers[i])

    return q_filt,a_filt


'''
Given a set of questions and answers (already tokenized), it converts words to indexes and adds a zero-padding if
necessary.

Returns one set of questions and another one of answers ater being processed.
'''


def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)


    return idx_q, idx_a


'''
Given a sequence, a lookup table and the expected length of the result, it replaces laces words with indices.

Returns a list of indexes that form the sequence.
'''

def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK]) # If the word is not in the vocabulary
    return indices + [0] * (maxlen - len(seq))

'''
Given a set of sentences, it gets the VOCAB_SIZE more common words and creates structures to easily map words to their ID.

Returns dictionaries index2word and word2index to map from word to index and viceversa.
'''
def index_(tokenized_sentences, freq_dist = None):
    # get frequency distribution
    if freq_dist is None:
        freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(VOCAB_SIZE)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

'''
Dataset "Cornell Movie Dialogs" contains too many sentences "I don't know" that highly affects the performance of the
bot. This method eliminates most of those sentences.
'''
def eliminate_dont_know(data):
    q_tmp = []
    a_tmp = []
    for i in range(0, int(len(data)/2)):
        rand = random.random()
        if 'I don\'t know' not in data[i] and 'I don\'t know' not in data[i*2] and rand < 0.95:
            q_tmp.append(data[i])
            a_tmp.append(data[i*2])
    return np.concatenate((np.array(q_tmp), np.array(a_tmp)), axis=0)


'''
Prepares the data before we can feed it to the Seq2Seq model.

It first loads the datasets and performs a basic preprocessing on their sentences. Then it selects and prepares all
those sentences that can be used in our Seq2Seq model.

Stores the final result to disk.

'''
def preprocess_data():

    print("Loading The Simpsons Dataset...")

    try:
        data_simp = load_simpsons_from_file()
    except FileNotFoundError:
        data_simp = load_simpsons(output_file=True)

    print("Basic preprocessing The Simpsons Dataset...")

    q_simp = data_simp.question
    q_simp_tok = basic_preproc(q_simp)
    a_simp = data_simp.answer
    a_simp_tok = basic_preproc(a_simp)

    print("Loading the Cornell Movies Dataset...")

    try:
        data_corn = load_cornell_from_file()
    except FileNotFoundError:
        data_corn = load_cornell(output_file=True)

    print("Basic preprocessing the Cornell Movies Dataset...")

    q_corn = data_corn.question
    q_corn_tok = basic_preproc(q_corn)
    a_corn = data_corn.answer
    a_corn_tok = basic_preproc(a_corn)

    # Filter by size
    print("Filtering sentences by size...")
    q_simp_filt, a_simp_filt = filter_data(q_simp_tok, a_simp_tok)
    all_simp_filt = np.concatenate((np.array(q_simp_filt), np.array(a_simp_filt)), axis=0)
    q_corn_filt, a_corn_filt = filter_data(q_corn_tok, a_corn_tok)
    all_corn_filt = np.concatenate((np.array(q_corn_filt), np.array(a_corn_filt)), axis=0)
    all_corn_filt = eliminate_dont_know(all_corn_filt)
    # All the dataset filtered
    all_filt = np.concatenate((np.array(all_corn_filt), np.array(all_simp_filt)), axis=0)

    # get frequency distribution
    print("Genereting word indexes and frequency distribution...")
    idx2w, w2idx, freq_dist = index_(all_filt)

    print("Applying zero padding...")
    idx_simp_q, idx_simp_a = zero_pad(q_simp_filt, a_simp_filt, w2idx)
    idx_corn_q, idx_corn_a = zero_pad(q_corn_filt, a_corn_filt, w2idx)


    print('Saving information to disk')
    print('Total number of question-answer cases for the Cornell Movies dataset: {}'.format(len(idx_corn_q)))
    print('Total number of question-answer cases for The Simpsons dataset: {}'.format(len(idx_simp_q)))
    if not os.path.exists(path_to_variables):
        os.makedirs(path_to_variables)

    # save them
    np.save(path_to_variables+'idx_simp_q.npy', idx_simp_q)
    np.save(path_to_variables+'idx_simp_a.npy', idx_simp_a)
    np.save(path_to_variables+'idx_corn_q.npy', idx_corn_q)
    np.save(path_to_variables+'idx_corn_a.npy', idx_corn_a)

    # let us now save the necessary dictionaries
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    # write to disk : data control dictionaries
    with open(path_to_variables+'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    return idx_simp_q, idx_simp_a, idx_corn_q, idx_corn_a, metadata

preprocess_data()
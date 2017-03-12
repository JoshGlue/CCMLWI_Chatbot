import nltk
import itertools
import numpy as np
import pickle
from load_data import load_cornell, load_cornell_from_file, load_simpsons_from_file,load_simpsons
import os
import re

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

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  if isinstance(sentence, float):   # In case the sentence is just a number.
      sentence = str(sentence)
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)
'''
def filter_line(line):
    return ''.join([ ch for ch in line if ch in WHITELIST ])

'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )
'''
def filter_data(sequences):
    q_filt = []
    a_filt = []
    for i in range(0, int(sequences.size/2)):
        length1 = len(sequences[i])
        length2 = len(sequences[i*2])
        if length1 >= limit['minq'] and length1 <= limit['maxq']:
            if length2 >= limit['mina'] and length2 <= limit['maxa']:
                q_filt.append(sequences[i])
                a_filt.append(sequences[i*2])

    return q_filt,a_filt


'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

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
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''

def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0] * (maxlen - len(seq))

'''
 Read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )
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


def preprocess_data():

    print("Loading The Simpsons Dataset...")

    try:
        data_simp = load_simpsons_from_file()
    except FileNotFoundError:
        data_simp = load_simpsons(output_file=True)

    print("Preprocessing The Simpsons Dataset...")

    q_simp = data_simp.question
    # All sentences to lower case
    q_simp = [line.lower() for line in q_simp]
    # Eliminate characters that are not in the white list.
    q_simp = [ filter_line(line) for line in q_simp ]
    a_simp = data_simp.answer
    a_simp = [line.lower() for line in a_simp]
    a_simp = [ filter_line(line) for line in a_simp ]
    q_simp_tok = [basic_tokenizer(line) for line in q_simp]
    a_simp_tok = [basic_tokenizer(line) for line in a_simp]
    all_simp_tok = np.concatenate((np.array(q_simp_tok), np.array(a_simp_tok)), axis=0)

    print("Loading the Cornell Movies Dataset...")

    try:
        data_corn = load_cornell_from_file()
    except FileNotFoundError:
        data_corn = load_cornell(output_file=True)

    print("Preprocessing the Cornell Movies Dataset...")

    q_corn = data_corn.question
    q_corn = [str(line).lower() for line in q_corn]
    q_corn = [ filter_line(line) for line in q_corn ]
    a_corn = data_corn.answer
    a_corn = [str(line).lower() for line in a_corn]
    a_corn = [ filter_line(line) for line in a_corn ]
    q_corn_tok = [basic_tokenizer(line) for line in q_corn]
    a_corn_tok = [basic_tokenizer(line) for line in a_corn]
    all_corn_tok = np.concatenate((np.array(q_corn_tok), np.array(a_corn_tok)), axis=0)

    # Filter by size
    print("Filtering sentences by size...")
    q_simp_filt, a_simp_filt = filter_data(all_simp_tok)
    all_simp_filt = np.concatenate((np.array(q_simp_filt), np.array(a_simp_filt)), axis=0)
    q_corn_filt, a_corn_filt = filter_data(all_corn_tok)
    all_corn_filt = np.concatenate((np.array(q_corn_filt), np.array(a_corn_filt)), axis=0)
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
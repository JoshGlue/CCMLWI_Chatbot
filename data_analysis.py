#
# Data Analysis before preprocessing
#


import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
import numpy as np
import re
import pandas as pd
from load_data import load_cornell, load_cornell_from_file, load_simpsons_from_file,load_simpsons
from collections import Counter
import math

cornell_lines_path = "./data/movie_lines.txt"
cornell_conversations = "./data/movie_conversations.txt"
simpsons_path = "./data/simpsons_script_lines.csv"

# Count for the length of the sequences.

num_bins = 30
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def main():
    # try:
    #     dataset = load_cornell_from_file()
    # except FileNotFoundError:
    #     dataset = load_cornell(output_file=True)
    #
    # questions = dataset.question
    # answers = dataset.answer
    # target_lengths = [ len(basic_tokenizer(line)) for line in answers]
    # source_lengths = [ len(basic_tokenizer(line)) for line in questions]
    #
    # #if FLAGS.plot_histograms:
    # plotHistoLengths("target lengths", target_lengths)
    # plotHistoLengths("source_lengths", source_lengths)
    # plotScatterLengths("target vs source length", "source length", "target length", source_lengths, target_lengths)


    ############################################

    try:
        dataset = load_simpsons_from_file()
    except FileNotFoundError:
        dataset = load_simpsons(output_file=True)

    questions = dataset.question
    answers = dataset.answer
    target_lengths = [len(basic_tokenizer(line)) for line in answers]
    source_lengths = [len(basic_tokenizer(line)) for line in questions]


    # if FLAGS.plot_histograms:
    plotHistoLengths("target lengths", target_lengths)
    plotHistoLengths("source_lengths", source_lengths)
    plotScatterLengths("target vs source length", "source length", "target length", source_lengths, target_lengths)

    count_word_frequency()

def plotScatterLengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	#plt.ylim(0, max(y_lengths))
	#plt.xlim(0,max(x_lengths))
	plt.ylim(0, 200)
	plt.xlim(0, 200)
	plt.show()

def plotHistoLengths(title, lengths):
	x = np.array(lengths)
	plt.hist(x,  num_bins, alpha=0.5)
	plt.title(title)
	plt.xlabel("Length")
	plt.ylabel("Number of Sequences")
	plt.xlim(0,80)
	plt.show()

def count_word_frequency():
    try:
        dataset = load_simpsons_from_file()
    except FileNotFoundError:
        dataset = load_simpsons(output_file=True)
    questions = dataset.question
    answers = dataset.answer
    raw_words = []
    for q,a in zip(questions, answers):
        q_words = str(q).split()
        a_words = str(a).split()
        for word in q_words:
            raw_words.append(word)
        for word in a_words:
            raw_words.append(word)

    print("Number of words: %d " % len(raw_words))
    simpsons_vocabulary = set(raw_words)
    corpus_vocab = Counter(raw_words)
    print("Size of vocabulary in the Simpsons: %d " % len(simpsons_vocabulary))

    # Get all the words the occur only once for the given wordlist
    desired_value = 1
    myDict = dict(Counter(raw_words))
    hapax_legomena_unsorted = [k for k, v in myDict.items() if v == desired_value]
    hapax_legomena = sorted(hapax_legomena_unsorted)
    print("Hapax legomena in the corpus: %d " % len(hapax_legomena))


if __name__=="__main__":
	main()


# Count for the frecuency of each word.



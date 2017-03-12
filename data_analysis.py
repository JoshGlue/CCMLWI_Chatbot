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
from preprocess_data import filter_line, basic_tokenizer

cornell_lines_path = "./data/movie_lines.txt"
cornell_conversations = "./data/movie_conversations.txt"
simpsons_path = "./data/simpsons_script_lines.csv"

# Count for the length of the sequences.

num_bins = range(1, 30)


def main():

    # print("Analysing The Simpsons dataset...")
    print("Loading The Simpsons Dataset...")
    try:
        dataset_simp = load_simpsons_from_file()
    except FileNotFoundError:
        dataset_simp = load_simpsons(output_file=True)

    questions_simp = dataset_simp.question
    answers_simp = dataset_simp.answer
    # target_lengths = [ len(basic_tokenizer(line)) for line in answers_simp]
    # source_lengths = [ len(basic_tokenizer(line)) for line in questions_simp]
    #
    # plotHistoLengths("target lengths", target_lengths)
    # plotHistoLengths("source_lengths", source_lengths)
    # plotScatterLengths("target vs source length", "source length", "target length", source_lengths, target_lengths)


    ############################################
    # print("Analysing Cornell Movies dataset...")

    print("Loading the Cornell Movies Dataset...")
    try:
        dataset_corn = load_cornell_from_file()
    except FileNotFoundError:
        dataset_corn = load_cornell(output_file=True)

    questions_corn = dataset_corn.question
    answers_corn = dataset_corn.answer
    # target_lengths = [len(basic_tokenizer(line)) for line in answers_corn]
    # source_lengths = [len(basic_tokenizer(line)) for line in questions_corn]
    #
    # plotHistoLengths("target lengths", target_lengths)
    # plotHistoLengths("source_lengths", source_lengths)
    # plotScatterLengths("target vs source length", "source length", "target length", source_lengths, target_lengths)

    ############################################

    print("Statistics for both datasets together...")
    all_q = np.concatenate((np.array(questions_simp), np.array(questions_corn)), axis=0)
    all_a = np.concatenate((np.array(answers_simp), np.array(answers_corn)), axis=0)
    all_data = np.concatenate((all_q, all_a), axis=0)

    target_lengths = [len(basic_tokenizer(line)) for line in all_a]
    source_lengths = [len(basic_tokenizer(line)) for line in all_q]

    plotHistoLengths("Answer lengths", target_lengths)
    plotHistoLengths("Question_lengths", source_lengths)
    plotScatterLengths("Answer vs Question length", "Question length", "Answer length", source_lengths, target_lengths)


    print("Calculating word frecuency...")
    count_word_frequency(all_data)

def plotScatterLengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)

	plt.ylim(0, 200)
	plt.xlim(0, 200)
	plt.show()

def plotHistoLengths(title, lengths):
	x = np.array(lengths)
	plt.hist(x,  bins=num_bins, rwidth=1)
	plt.title(title)
	plt.xlabel("Length")
	plt.ylabel("Number of Sequences")
	plt.xlim(0,80)
	plt.show()

def count_word_frequency(dataset):

    raw_words = []
    print(">> Letters to lower case and eliminate characters in the black list...")
    for line in dataset:
        # All sentences to lower case
        line = str(line).lower()
        # Eliminate characters that are not in the white list.
        line = filter_line(line)
        line = str(line).split()
        for word in line:
            raw_words.append(word)

    print("Total number of words: %d " % len(raw_words))
    vocabulary = set(raw_words)
    print("Size of vocabulary: %d " % len(vocabulary))

    # Get all the words the occur only once for the given wordlist
    desired_value = 3
    myDict = dict(Counter(raw_words))
    hapax_legomena_unsorted = [k for k, v in myDict.items() if v <= desired_value]
    hapax_legomena = sorted(hapax_legomena_unsorted)
    print("Words that appear 3 times or less: %d " % len(hapax_legomena))


if __name__=="__main__":
	main()



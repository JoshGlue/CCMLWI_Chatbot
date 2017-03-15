#
# Data Analysis of both datasets.
#
# This code is not called by any other script, but it was used to determine some basic characteristics of the datasets.
# It tests how long the sentences are (since we will need to specify a maximum length for the seq2seq) and how big and
# frecuent our vocabulary is.
#


from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from preproc.preprocess_data import filter_line, basic_tokenizer

from preproc.load_data import load_cornell, load_cornell_from_file, load_simpsons_from_file,load_simpsons


# Count for the length of the sequences.

num_bins = range(1, 50)


def main():

    # print("Analysing The Simpsons dataset...")
    print("Loading The Simpsons Dataset...")
    try:
        dataset_simp = load_simpsons_from_file()
    except FileNotFoundError:
        dataset_simp = load_simpsons(output_file=True)

    questions_simp = dataset_simp.question
    answers_simp = dataset_simp.answer
    target_lengths = [ len(basic_tokenizer(line)) for line in answers_simp]
    source_lengths = [ len(basic_tokenizer(line)) for line in questions_simp]

    plotHistoLengths("Answers length - The Simpsons", target_lengths)
    plotHistoLengths("Questions lengths  - The Simpsons", source_lengths)
    plotScatterLengths("Answers vs. Questions length - The Simpsons", "Question length", "Answer length", source_lengths, target_lengths)


    ############################################
    # print("Analysing Cornell Movies dataset...")

    print("Loading the Cornell Movies Dataset...")
    try:
        dataset_corn = load_cornell_from_file()
    except FileNotFoundError:
        dataset_corn = load_cornell(output_file=True)

    questions_corn = dataset_corn.question
    answers_corn = dataset_corn.answer
    target_lengths = [len(basic_tokenizer(line)) for line in answers_corn]
    source_lengths = [len(basic_tokenizer(line)) for line in questions_corn]

    plotHistoLengths("Answers length - Cornell Movies", target_lengths)
    plotHistoLengths("Questions length - Cornell Movies", source_lengths)
    plotScatterLengths("Answers vs. Questions length - Cornell Movies", "Question length", "Answer length", source_lengths, target_lengths)

    ############################################

    print("Statistics for both datasets together...")
    all_q = np.concatenate((np.array(questions_simp), np.array(questions_corn)), axis=0)
    all_a = np.concatenate((np.array(answers_simp), np.array(answers_corn)), axis=0)
    all_data = np.concatenate((all_q, all_a), axis=0)

    target_lengths = [len(basic_tokenizer(line)) for line in all_a]
    source_lengths = [len(basic_tokenizer(line)) for line in all_q]

    plotHistoLengths("Answer lengths - The Simpsons and Cornell Movies", target_lengths)
    plotHistoLengths("Question lengths - The Simpsons and Cornell Movies", source_lengths)
    plotScatterLengths("Answer vs Question length - The Simpsons and Cornell Movies", "Question length", "Answer length", source_lengths, target_lengths)


    print("Calculating word frecuency...")
    count_word_frequency(all_data)

def plotScatterLengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)

	plt.ylim(0, 150)
	plt.xlim(0, 150)
	plt.show()

def plotHistoLengths(title, lengths):
	x = np.array(lengths)
	plt.hist(x,  bins=num_bins, rwidth=2)
	plt.title(title)
	plt.xlabel("Length")
	plt.ylabel("Number of Sequences")
	plt.xlim(0,50)
	plt.show()

def count_word_frequency(dataset):

    raw_words = []
    print("Preprocessing the sentences...")
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
    desired_value = 2
    hapax_legomena_unsorted = [k for k, v in myDict.items() if v <= desired_value]
    hapax_legomena = sorted(hapax_legomena_unsorted)
    print("Words that appear 2 times or less: %d " % len(hapax_legomena))
    desired_value = 1
    hapax_legomena_unsorted = [k for k, v in myDict.items() if v <= desired_value]
    hapax_legomena = sorted(hapax_legomena_unsorted)
    print("Words that appear 2 times or less: %d " % len(hapax_legomena))

if __name__=="__main__":
	main()



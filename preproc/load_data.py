import pandas as pd
import os

SIMPSONS_PATH = "./data/simpsons_script_lines.csv"
CORNELL_LINES_PATH = "./data/movie_lines.txt"
CORNELL_CONVERSATIONS_PATH = "./data/movie_conversations.txt"
CORNELL_DELIMITER = " +++$+++ "
OUT_PATH = "./usable_data/"
OUT_SIMP_PATH = OUT_PATH + "homer_context.csv"
OUT_CORN_PATH = OUT_PATH + "cornell_q_a.csv"
CHARACTER_ID = 2    # 2 Is the character ID of Homer Simpson.

#####################################
# The Simpsons Dataset
#####################################

'''
Given the original file of the dataset "The Simpsons by the Data", it gets and returns all the sentences said by
CHARACTER_ID (that will be marked as answers) together with the line of dialog preceding that sentence (questions).

If output_file is 'True', then the new data is stored in a file so we do not have to perform this operation again.

Original dataset: https://www.kaggle.com/wcukierski/the-simpsons-by-the-data
'''

def load_simpsons(output_file=False):
    context_distance = 1
    row_name = 'spoken_words'

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    df = pd.read_csv(SIMPSONS_PATH, delimiter=',', error_bad_lines=False)
    df_homer = df[df['character_id'] == CHARACTER_ID]
    df_context = pd.DataFrame(columns=["question", "answer"])
    i = 0
    print('Loading Simpsons Dataset')
    for index, row in df_homer.iterrows():
        if i % 1000 == 0:
            print("Progress: {}%\r".format(100 * (i / float(len(df_homer.index)))))
        line = str(row[row_name])
        if not line == 'nan':
            context_range = range(index - context_distance, index)
            context = ["", row[row_name]]
            for ix in context_range:
                context_positive = str(df.loc[ix][row_name])
                if not context_positive == "nan":
                    context[0] += context_positive + " "
                else:
                    context = ''
            if not context == '':
                df_context.loc[i] = context
        i += 1
    print('Loading Simpsons Dataset - Done')

    if output_file:
        df_context.to_csv(OUT_SIMP_PATH)

    return df_context

"Loads ans already existing file that stores pairs of Question-Answers"

def load_simpsons_from_file():
    return pd.read_csv(OUT_SIMP_PATH)


#####################################
# Cornell Movies Dataset
#####################################

'''
    From the "movie_lines.txt" of the original 'Cornell Movie Dialogs Dataset', it gets all the ID and spoken dialog of
    the conversations.
'''
def _get_id_and_lines():
    lines = open(CORNELL_LINES_PATH).read().split('\n')
    id_and_lines = {}
    for line in lines:
        split_line = line.split(CORNELL_DELIMITER)
        if len(split_line) == 5:    # The last line is an empty line
            id_and_lines[split_line[0]] = split_line[4]
    return id_and_lines


'''
    1. From 'movie_conversations.txt' of the original 'Cornell Movie Dialogs Dataset', it gets all the conversations of
    the dataset (each sentence is represented by the ID of the spoken sentence).
'''

def _get_convs():
    conv_lines = open(CORNELL_CONVERSATIONS_PATH).read().split('\n')
    convs = []
    for line in conv_lines[:-1]:
        split_line = line.split(CORNELL_DELIMITER)[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(split_line.split(','))
    return convs


'''
Given the original file of the dataset "Cornell Movie Dialogs Dataset", it puts together all the possible conversations
and creates pairs of Question-Answer where the 'question' is the line immediately after an 'answer'.

If output_file is 'True', then the new data is stored in a file so we do not have to perform this operation again.

Original dataset: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
'''

def load_cornell(output_file=False):

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    id_and_lines = _get_id_and_lines()
    convs = _get_convs()

    q_and_a = []

    j = 0
    k = 0
    for conv in convs:
        for i in range(0,len(conv)-1):
            line1 = id_and_lines[conv[i]]
            line2 = id_and_lines[conv[i+1]]

            q_and_a.append([line1, line2])
            k+=1
        j+=1
    if i % 1000 == 0:
        print("Progress: {}%".format(100 * (j / float(len(convs)))))
    df_q_and_a = pd.DataFrame(q_and_a, columns=['question', 'answer'])
    if output_file:
        df_q_and_a.to_csv(OUT_CORN_PATH)

    return df_q_and_a

"Loads ans already existing file that stores pairs of Question-Answers"

def load_cornell_from_file():
    return pd.read_csv(OUT_CORN_PATH)

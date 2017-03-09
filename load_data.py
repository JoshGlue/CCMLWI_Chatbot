import pandas as pd
from random import randint


SIMPSONS_PATH = "./data/simpsons_script_lines.csv"
CORNELL_LINES_PATH = "./data/movie_lines.txt"
CORNELL_CONVERSATIONS_PATH = "./data/movie_conversations.txt"
CORNELL_DELIMITER = " +++$+++ "
OUT_SIMP_PATH = "./homer_context.csv"
OUT_CORN_PATH = "./cornell_q_a.csv"

def load_simpsons(output_file=False):
    homer_id = 2
    context_distance = 1
    row_name = 'spoken_words'

    df = pd.read_csv(SIMPSONS_PATH, delimiter=',', error_bad_lines=False)
    df_homer = df[df['character_id'] == homer_id]
    df_context = pd.DataFrame(columns=["question", "answer", "Label"])
    i = 0
    j=0
    print('Loading Simpsons Dataset')
    for index, row in df_homer.iterrows():
        line = str(row[row_name])
        if not line=='nan':
            j += 1
            print("Progress: {}%".format(100*(j/float(len(df_homer.index)))))
            context_range = range(index - context_distance, index)
            context_positive = ["", row[row_name], 1]
            context_negative = ["", row[row_name],0]
            for ix in context_range:
                context_positive_sentence = str(df.loc[ix][row_name])
                if not context_positive_sentence == "nan":
                    context_positive[0] += context_positive_sentence +  " "
                else:
                    context_positive = ''
                context_negative_sentence = str(df.loc[randint(0,len(df.index)-1)][row_name])
                if not context_negative_sentence == "nan":
                    context_negative[0] += context_negative_sentence + " "
                context_negative[0] += str(df.loc[randint(0,len(df.index)-1)][row_name]) + " "
            if not context_positive == '':
                df_context.loc[i] = context_positive
            df_context.loc[i] = context_negative
        i+= 2
    print('Loading Simpsons Dataset - Done')

    if output_file:
        df_context.to_csv(OUT_SIMP_PATH)

    return df_context

def load_simpsons_from_file():
    return pd.read_csv(OUT_SIMP_PATH)


def _get_id_and_lines():
    lines = open(CORNELL_LINES_PATH).read().split('\n')
    id_and_lines = {}
    for line in lines:
        split_line = line.split(CORNELL_DELIMITER)
        if len(split_line) == 5:    # The last line is an empty line
            id_and_lines[split_line[0]] = split_line[4]
    return id_and_lines


'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''

def _get_convs():
    conv_lines = open(CORNELL_CONVERSATIONS_PATH).read().split('\n')
    convs = []
    for line in conv_lines[:-1]:
        split_line = line.split(CORNELL_DELIMITER)[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(split_line.split(','))
    return convs


def load_cornell(output_file=False):
    out_corn_path = "./cornell_q_a.csv"

    print('Loading Cornell Movies Dataset')
    id_and_lines = _get_id_and_lines()
    convs = _get_convs()

    df_q_and_a = pd.DataFrame(columns=["Question", "Answer"])
    q_and_a = []

    j = 0
    k = 0
    for conv in convs:
        print("Progress: {}%".format(100*(j/float(len(convs)))))
        for i in range(0,len(conv)-1):
            line1 = id_and_lines[conv[i]]
            line2 = id_and_lines[conv[i+1]]

            q_and_a.append([line1, line2])
            k+=1
        j+=1

    print("Progress: {}%".format(100 * (j / float(len(convs)))))
    df_q_and_a = pd.DataFrame(q_and_a, columns=['question', 'answer'])
    if output_file:
        df_q_and_a.to_csv(out_corn_path)

    print('Loading the Cornell Movies Dataset - Done')
    return df_q_and_a

def load_cornell_from_file():
    return pd.read_csv(OUT_CORN_PATH)

#load_simpsons(output_file=True)
#load_cornell(output_file=True)
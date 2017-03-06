import pandas as pd
from random import randint
HOMER = 2
CONTEXT_DISTANCE = 6
df = pd.read_csv('./data/simpsons_script_lines.csv', delimiter=',', error_bad_lines=False)
df_homer = df[df['character_id'] == HOMER]
df_context = pd.DataFrame(columns=["Context", "Utterance", "Label"])
i = 0
j=0
for index, row in df_homer.iterrows():
    j += 1
    print("Progress: {}%".format(100*(j/float(len(df_homer.index)))))
    context_range = range(index - CONTEXT_DISTANCE, index-1)
    context_positive = ["" , row["normalized_text"], 1]
    context_negative = ["", row["normalized_text"],0]
    for ix in context_range:
        context_positive_sentence = str(df.loc[ix]["normalized_text"])
        if not context_positive_sentence == "nan":
            context_positive[0] += context_positive_sentence +  " "
        context_negative_sentence = str(df.loc[randint(0,len(df.index)-1)]["normalized_text"])
        if not context_negative_sentence == "nan":
            context_negative[0] += context_negative_sentence + " "

        context_negative[0] += str(df.loc[randint(0,len(df.index)-1)]["normalized_text"]) + " "
    df_context.loc[i] = context_positive
    i += 1
    df_context.loc[i] = context_negative
    i+= 1
df_context.to_csv("./homer_context.csv")



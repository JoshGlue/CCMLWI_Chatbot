# Homer Simpson Chatbot
![Homer](http://www.3ders.org/images2015/mythbusters-3d-models-homer-simpson-new-experiment-00005.png)
We created a chatbot that tries to mimic the famous Homer Simpson. When a message is send to Homer, it responds with generated sentences based on script lines of the Simpsons.

## Data
There has been made use of two datasets to enhance the dictionary of the embeddings that are used. The first dataset that is used is "The Simpsons by the Data" dataset, that can be found on Kaggle (https://www.kaggle.com/wcukierski/the-simpsons-by-the-data). This dataset consists of all locations, episodes, characters and the script lines. The latter is used for training our model. The second dataset that is used is the Cornell Movie-Dialogs Corpus, which can be found here: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html. This dataset consists of 300K utterances of different movies.


### Preprocessing
To train the model on both datasets the data needs to preprocessed, so that the model trains in the same fashion on both datasets.

The preprocessing that we chose are the following:
- Lowercase all characters
- Use a whitelist to filter all the unwanted characters. (whitelist: "0123456789abcdefghijklmnopqrstuvwxyz .?!")
- Cap the number of words in a sentence to 20.

Both datasets that we used had only spoken lines in the dataset, so there was no need to do special preprocessing steps to remove exclamations or something.


The data needs to be put in in the following format:

**Question** | **Answer**
---|---
| 


The question is the sentence that precedes the answer that the chatbot needs to give when a chat line has been received. So it does not need to be a question per se.
All utterances of the cornell dataset are used for training. The Simpson dataset are filtered on sentences that homer speaks and these account for the answers. The sentence that precedes the sentence of homer is the question for the sentence pair.



## Training procedure
We first trained on the whole Cornell Movie-Dialogs Corpus for 15,000 epochs of size 32, which totalled in a size of 480,000 question-answer pairs that are trained on. We then trained on 5,000 epochs of size 32, which is in total 160,000 sentence pairs of Homer Simpson.


## Model
The model that we used to generate the sentences is a seq2seq model with a LTSM cell. By using a dual encoder the question get encoded in the LTSM cell and decoded in to an answer. This therefore generates sentences on the fly. For implementing the seq2seq model, we followed the tutorial that can be found on http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/. 

### Requirements:
- Python 3
- Tensorflow 0.12
- NLTK

Information about installing Tensorflow 0.12 can be found here:
https://www.tensorflow.org/versions/r0.12/get_started/os_setup


### Training the Chatbot
1. Clone this repository.
2. `cd` into the directory of the chatbot.
2. Download the Simpsons Dataset https://www.kaggle.com/wcukierski/the-simpsons-by-the-data and the Cornell Movie-Dialogs Corpus https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
3. Extract the archives and place `simpsons_script_lines.csv`, `movie_conversations.txt` and `movie_lines.txt` in `./data`
4. Run `python ./train.py`

### Running the Chatbot
1. Train the model as described in the previous section or download the pretrained model [LINK NEEDED](LINK NEEDED)
2. Run `python ./telegram.py`
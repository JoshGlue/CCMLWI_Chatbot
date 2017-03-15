# Homer Simpson Chatbot
We created a chatbot that tries to mimic the famous Homer Simpson. When a message is send to Homer, it responds with generated sentences based on script lines of the Simpsons.

## Data
There has been made use of two datasets to enhance the dictionary of the embeddings that are used. The first dataset that is used is "The Simpsons by the Data" dataset, that can be found on Kaggle (https://www.kaggle.com/wcukierski/the-simpsons-by-the-data). This dataset consists of all locations, episodes, characters and the script lines. The latter is used for training our model. The second dataset that is used is the Cornell Movie-Dialogs Corpus, which can be found here: http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip. This dataset consists of 300K utterances of different movies.


### Preprocessing
To train the model on both datasets the data needs to preprocessed, so that the model trains in the same fashion on both datasets. Both datasets that we used had only spoken lines in the dataset, so a 

We chose to do the following preprocessing steps:
- Lowercase all characters
- Use a whitelist to filter all the unwanted characters. (whitelist: "0123456789abcdefghijklmnopqrstuvwxyz .?!")
- Cap the number of words in a sentence to 20.

The data needs to be put in in the following format:

**Question** | **Answer**
---|---
| 
The question is the sentence that precedes the answer that the chatbot needs to give when a chat line has been received. So it does not need to be a question per se.

## Training procedure
We first trained on the whole Cornell Movie-Dialogs Corpus for 15,000 epochs of size 32, which totalled in a size 

The model that we used to generate the sentences is a seq2seq model with a LTSM kernel. The implementation of the seq2seq model is the one that can be found at http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/. 

### Requirements:
- Python 3
- Tensorflow 0.12
- NLTK

Information about installing Tensorflow 0.12 can be found here:
https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/g3doc/get_started/os_setup.md


### Training the chatbot



``` bash
cd seq2seq/ckpt/twitter/
./pull
tar xzf seq2seq_twitter_1024x3h_i43000.tar.gz
cd ../../
cd datasets/twitter/
./pull
tar xzf seq2seq.twitter.tar.gz
cd ../../../
python telegram.py
```

# CCMLWI_Chatbot

Prequisites:
Python 3
Tensorflow 0.12

cd seq2seq/ckpt/twitter/
./pull 

tar xzf seq2seq_twitter_1024x3h_i43000.tar.gz

# this will take some time ~830 MB

cd ../../
# then you need the datasets to test the model

cd datasets/twitter/
./pull # this wont take long

tar xzf seq2seq.twitter.tar.gz

cd ../../../

python telegram.py
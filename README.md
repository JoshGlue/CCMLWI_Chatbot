# CCMLWI_Chatbot

### Prequisites:
- Python 3
- Tensorflow 0.12


### Running the chatbot
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

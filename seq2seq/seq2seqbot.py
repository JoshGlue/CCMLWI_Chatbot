from . import seq2seq_wrapper, data_utils 
from .datasets.twitter import data
import numpy as np
import pickle
metadata, idx_q, idx_a = data.load_data(PATH='seq2seq/datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='seq2seq/ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )
sess = model.restore_last_session()
def send_message(text, chat_id):
	text = text.lower()
	with open('seq2seq/datasets/twitter/metadata.pkl', 'rb') as f:
		metadata = pickle.load(f)
	text = data.filter_line(text, data.EN_WHITELIST)
	text_tokenized = text.split(' ')
	idx2w, w2idx, freq_dist = data.index_([text_tokenized], vocab_size=data.VOCAB_SIZE, freq_dist=metadata['freq_dist'])
	idx_q, _ = data.zero_pad([text_tokenized], [text_tokenized], w2idx)
	output = model.predict(sess, idx_q.T)
	decoded = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ').split(' ')
	return ' '.join(decoded)
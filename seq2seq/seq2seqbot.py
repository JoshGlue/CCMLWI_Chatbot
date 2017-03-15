
from . import seq2seq_wrapper, data_utils 
import numpy as np
import pickle

from preproc import preprocess_data as prep_data
from . import seq2seq_wrapper, data_utils

'''
Loads and returns the metadata used for the training of our Chatbox.
'''
def load_model_metadata(PATH='variables/'):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return metadata

metadata = load_model_metadata()
# parameters 
xseq_len = 20
yseq_len = 20
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
path_model_ckpt = './ckpt/model/'


model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_model_ckpt,
                               emb_dim=emb_dim,
                               num_layers=3
                               )
sess = model.restore_last_session()


'''
Given a sentence, it feeds it to the Chatbox seq2seq model and outputs its response.
'''
def send_message(text, chat_id):
	text = text.lower()
	text = prep_data.filter_line(text)
	text_tokenized = text.split(' ')
	idx2w, w2idx, freq_dist = prep_data.index_([text_tokenized],freq_dist=metadata['freq_dist'])
	idx_q, _ = prep_data.zero_pad([text_tokenized], [text_tokenized], w2idx)
	output = model.predict(sess, idx_q.T)
	decoded = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ').split(' ')
	return ' '.join(decoded)
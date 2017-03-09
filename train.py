
# In[1]:

import tensorflow as tf
import numpy as np

# preprocessed data
from seq2seq.datasets.twitter import data
import seq2seq.data_utils as data_utils
import pickle

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def load_data(PATH='', concatenate=False, multiple_simpson=5):
	# When concatenate is true, then the simpson and cornell datasets will be merged.
	# The multiple_simpson parameter determines how many time the simpson dataset need to be concatenated
    # read data control dictionaries
    with open(PATH + 'seq2seq/datasets/twitter/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_simp_q.npy')
    idx_a = np.load(PATH + 'idx_simp_a.npy')

    if concatenate:
    	idx_cornell_q = np.load(PATH + 'idx_corn_q.npy')
    	idx_cornell_a = np.load(PATH + 'idx_corn_q.npy')
    	arrays_q = [idx_cornell_q]
    	arrays_a = [idx_cornell_a]
    	for i in range(multiple_simpson):
    		arrays_q.append(idx_q)
    		arrays_a.append(idx_a)

    	idx_q = np.concatenate(arrays_q)
    	idx_a = np.concatenate(arrays_a)
    	idx_q, idx_a = unison_shuffled_copies(idx_q, idx_a)
    return metadata, idx_q, idx_a


# load data from pickle and npy files
metadata, idx_q, idx_a = load_data(PATH='')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

from seq2seq import seq2seq_wrapper


# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)

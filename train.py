
# In[1]:

import tensorflow as tf
import numpy as np
import os
# preprocessed data
from seq2seq.datasets.twitter import data
import seq2seq.data_utils as data_utils
import pickle
from preprocess_data import preprocess_data

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_simp_data(PATH='./variables/'):
    # When concatenate is true, then the simpson and cornell datasets will be merged.
    # The multiple_simpson parameter determines how many time the simpson dataset need to be concatenated
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # read numpy arrays
    idx_simp_q = np.load(PATH + 'idx_simp_q.npy')
    idx_simp_a = np.load(PATH + 'idx_simp_a.npy')

    return metadata, idx_simp_q, idx_simp_a



def load_corn_data(PATH='./variables/'):
	# When concatenate is true, then the simpson and cornell datasets will be merged.
	# The multiple_simpson parameter determines how many time the simpson dataset need to be concatenated
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # read numpy arrays
    idx_corn_q = np.load(PATH + 'idx_corn_q.npy')
    idx_corn_a = np.load(PATH + 'idx_corn_a.npy')

    return metadata, idx_corn_q, idx_corn_a


preprocess_data()

# Start training with the whole movies dataset
metadata, idx_q, idx_a = load_corn_data()
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
path_corn_ckpt = './ckpt/cornell/'
path_corn_ckpt = './ckpt/homer/'


from seq2seq import seq2seq_wrapper

if not os.path.exists(path_corn_ckpt):
    os.makedirs(path_corn_ckpt)


# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_corn_ckpt,
                               emb_dim=emb_dim,
                               num_layers=3
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen, sess)


# parameters

metadata, idx_q, idx_a = load_simp_data()
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024
path_simp_ckpt = 'cornell/homer/'

from seq2seq import seq2seq_wrapper

if not os.path.exists(path_simp_ckpt):
    os.makedirs(path_simp_ckpt)

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_simp_ckpt,
                               emb_dim=emb_dim,
                               num_layers=3
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
import numpy as np
import os
import seq2seq.data_utils as data_utils
import pickle
from preprocess_data import preprocess_data

'''
Loads the already preprocessed data about 'The Simpsons by the Data' dataset.
'''
def load_simp_data(PATH='./variables/'):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # read numpy arrays
    idx_simp_q = np.load(PATH + 'idx_simp_q.npy')
    idx_simp_a = np.load(PATH + 'idx_simp_a.npy')

    return metadata, idx_simp_q, idx_simp_a


'''
Loads the already preprocessed data about 'Cornell Movies Dialogs' dataset.
'''
def load_corn_data(PATH='./variables/'):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # read numpy arrays
    idx_corn_q = np.load(PATH + 'idx_corn_q.npy')
    idx_corn_a = np.load(PATH + 'idx_corn_a.npy')

    return metadata, idx_corn_q, idx_corn_a


# Start training with the whole movies dataset

preprocess_data()

metadata, idx_q, idx_a = load_corn_data()
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024
path_model_ckpt = './ckpt/model/'

from seq2seq import seq2seq_wrapper


# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_model_ckpt,
                               emb_dim=emb_dim,
                               epochs = 15000,
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

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_model_ckpt,
                               emb_dim=emb_dim,
                                epochs = 5000,
                               num_layers=3
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
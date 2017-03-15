import pickle

import numpy as np

import seq2seq.data_utils as data_utils
from preproc.preprocess_data import preprocess_data
from seq2seq import seq2seq_wrapper

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

###########################
# Start training the model with the "Cornell Movies Dialog Dataset"
###########################

# Read metadata, questions and answers
try:
    metadata, idx_q, idx_a = load_corn_data()
except FileNotFoundError:
    _,_,idx_q, idx_a, metadata = preprocess_data()

# Divide the dataset in training, test and validation sets.

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

xseq_len = trainX.shape[-1] # Length of the input
yseq_len = trainY.shape[-1] # Length of the output
batch_size = 16
xvocab_size = len(metadata['idx2w']) # Vocabulary size of the input
yvocab_size = xvocab_size # Vocabulary size of the output
emb_dim = 1024 # Embedding size
path_model_ckpt = './ckpt/model/'   # Path where we will store the model

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_model_ckpt,
                               emb_dim=emb_dim,
                               epochs = 15000,
                               num_layers=3
                               )
# Create batches
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
# Train
sess,_ = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen, sess)

###########################
# Start training the model with the "Cornell Movies Dialog Dataset"
###########################

try:
    metadata, idx_q, idx_a = load_simp_data()
except FileNotFoundError:
    idx_q, idx_a,_,_, = preprocess_data()

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

xseq_len = trainX.shape[-1] # Length of the input
yseq_len = trainY.shape[-1] # Length of the output
batch_size = 16
xvocab_size = len(metadata['idx2w']) # Vocabulary size of the input
yvocab_size = xvocab_size # Vocabulary size of the output
emb_dim = 1024 # Embedding size
path_model_ckpt = './ckpt/model/'   # Path where we will store the model

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=path_model_ckpt,
                               emb_dim=emb_dim,
                                epochs = 5000,
                               num_layers=3
                               )
# Train
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
# Create batches
sess, _ = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
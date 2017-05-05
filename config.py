""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

This file contains the hyperparameters for the model.
"""

# parameters for processing the dataset
DATA_PATH = 'datasets'
REDDIT_PROMPT = 'questions.csv'
REDDIT_RESPONSE = 'answers.csv'
REDDIT_DATA = 'comments.txt'
OUTPUT_FILE = 'output_convo.txt'
TRAINING_RECORD_FILE = 'training_record.txt'
TESTING_RECORD_FILE = 'testing_record.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 3

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

# Size limit on the data
MIN_Q = 1
MAX_Q = 30
MIN_A = 1
MAX_A = 30

# Raw data size: 564,069 pairs of q&a
TWTR_LEN = 564069
TESTSET_SIZE = int(TWTR_LEN * 0.25)# 25% as Test Data

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz,.!?\' '
# BLACKLIST = ['http', 'https', ]
# model parameters
# [19530, 17449, 17585, 23444, 22884, 16435, 17085, 18291, 18931]
# BUCKETS = [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44)]

# [86202, 90338, 89196]
BUCKETS = [(9, 11), (14, 17), (20, 25)]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5 # Learning Rate
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 38691
DEC_VOCAB = 38536

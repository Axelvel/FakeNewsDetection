import torch as tf
import pandas as pd
from model import FakeNewsClassifier

DATASET = 'liar_dataset/'
TRAIN_PATH = DATASET + 'train.tsv'
TEST_PATH = DATASET + 'test.tsv'
EVAL_PATH = DATASET + 'valid.tsv'

COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely-true', 'false', 'half-true', 'mostly-true', 'pants-on-fire', 'context']

df_train = pd.read_csv(TRAIN_PATH, sep='\t')
df_test = pd.read_csv(TEST_PATH, sep='\t')
df_eval = pd.read_csv(EVAL_PATH, sep='\t')

# Setting the columns
df_train.columns = COLUMNS
df_test.columns = COLUMNS
df_eval.columns = COLUMNS

# Data preprocessing
df_train.pop('id')

print(df_train)
print(df_train.shape)

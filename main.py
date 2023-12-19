import torch as tf
import pandas as pd

DATASET = 'liar_dataset/'
TRAIN_PATH = DATASET + 'train.tsv'
TEST_PATH = DATASET + 'test.tsv'
EVAL_PATH = DATASET + 'valid.tsv'

df = pd.read_csv(TRAIN_PATH, sep='\t')

print(df)
print(df.shape)

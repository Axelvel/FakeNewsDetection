import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET = 'liar_dataset/'
TRAIN_PATH = DATASET + 'train.tsv'
TEST_PATH = DATASET + 'test.tsv'
EVAL_PATH = DATASET + 'valid.tsv'

COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'context']
LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

df_train = pd.read_csv(TRAIN_PATH, sep='\t', names=COLUMNS)
df_test = pd.read_csv(TEST_PATH, sep='\t', names=COLUMNS)
df_eval = pd.read_csv(EVAL_PATH, sep='\t', names=COLUMNS)

# Data preprocessing
df_train.pop('id')

# Displaying histogram

label_distribution = []
for label in LABELS:
    label_distribution.append(np.count_nonzero(df_train['label'] == label))

print(label_distribution)

plt.bar(LABELS, label_distribution)
plt.show()

print(df_train)
print(df_train.shape)
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

DATASET = 'liar_dataset/'
TRAIN_PATH = DATASET + 'train.tsv'
TEST_PATH = DATASET + 'test.tsv'
EVAL_PATH = DATASET + 'valid.tsv'

COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'context']
LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

df_train = pd.read_csv(TRAIN_PATH, sep='\t', names=COLUMNS)
df_test = pd.read_csv(TEST_PATH, sep='\t',names=COLUMNS)
df_eval = pd.read_csv(EVAL_PATH, sep='\t',names=COLUMNS)

# Data preprocessing
df_train.pop('id')

# Displaying histogram
label_distribution = []
for label in LABELS:
    label_distribution.append(np.count_nonzero(df_train['label'] == label))

print(label_distribution)
plt.bar(LABELS, label_distribution)
#plt.show()

# Encoding labels
label_encoder = LabelEncoder()
label_encoder.fit(LABELS)
df_train['label'] = label_encoder.transform(df_train['label'])
joblib.dump(label_encoder, 'data/label_encoder.plk')


print(df_train)
print(df_train.shape)
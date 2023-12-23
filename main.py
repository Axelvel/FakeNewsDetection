import numpy as np
import os
import preprocessing
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
df_eval.pop('id')
df_test.pop('id')

all_topic, nb_topic_max = preprocessing.extract_topic(df_train)
all_speaker = list(set(df_train["speaker"].to_list()))
all_job = list(set(df_train["job"].to_list()))

train_sentences, train_meta_data, train_mask = preprocessing.preprocess(df_train,all_topic, nb_topic_max,all_speaker,all_job)
eval_sentences, eval_meta_data, eval_mask = preprocessing.preprocess(df_eval,all_topic, nb_topic_max,all_speaker,all_job)
test_sentences, test_meta_data, test_mask = preprocessing.preprocess(df_test,all_topic, nb_topic_max,all_speaker,all_job)

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
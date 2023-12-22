import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATASET = 'liar_dataset/'
TRAIN_PATH = DATASET + 'train.tsv'
TEST_PATH = DATASET + 'test.tsv'
EVAL_PATH = DATASET + 'valid.tsv'

labels = ["pants-fire","false","barely-true","half-true","mostly-true","true"]

COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely-true', 'false', 'half-true', 'mostly-true', 'pants-on-fire', 'context']

df_train = pd.read_csv(TRAIN_PATH, sep='\t')
df_test = pd.read_csv(TEST_PATH, sep='\t')
df_eval = pd.read_csv(EVAL_PATH, sep='\t')

# Setting the columns
df_train.columns = COLUMNS
df_test.columns = COLUMNS
df_eval.columns = COLUMNS

#plotting data_repartition
if not os.path.isfile("./data_repartition.png"):
    label_data = np.array(df_train.label.str.split(expand=True))

    stats = []
    for word in labels:
        stats.append(np.count_nonzero(label_data == word))

    print(stats)

    plt.bar(labels, stats)
    plt.savefig("./data_repartition.png")

# Data preprocessing
df_train.pop('id')

print(df_train)
print(df_train.shape)

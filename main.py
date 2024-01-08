import numpy as np
import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from model import FakeNewsClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import time

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

DATASET = 'liar_dataset/'
TRAIN_PATH = DATASET + 'train.tsv'
TEST_PATH = DATASET + 'test.tsv'
EVAL_PATH = DATASET + 'valid.tsv'

COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'context']
LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']

df_train = pd.read_csv(TRAIN_PATH, sep='\t', names=COLUMNS)
df_train = df_train[:100] # For testing purposes
df_test = pd.read_csv(TEST_PATH, sep='\t', names=COLUMNS)
df_eval = pd.read_csv(EVAL_PATH, sep='\t', names=COLUMNS)

# Data preprocessing
df_train.pop('id')
df_eval.pop('id')
df_test.pop('id')

all_topic, nb_topic_max = preprocessing.extract_topic(df_train)
all_speaker = list(set(df_train["speaker"].to_list()))
all_job = list(set(df_train["job"].to_list()))

train_sentences, train_meta_data, train_mask = preprocessing.preprocess(df_train, all_topic, nb_topic_max, all_speaker, all_job)
eval_sentences, eval_meta_data, eval_mask = preprocessing.preprocess(df_eval, all_topic, nb_topic_max, all_speaker, all_job)
test_sentences, test_meta_data, test_mask = preprocessing.preprocess(df_test, all_topic, nb_topic_max, all_speaker, all_job)

train_a, train_b, train_c , train_d, train_e= torch.split(train_sentences,int(len(train_sentences)/5))
train_sentences = [train_a.to(device), train_b.to(device), train_c.to(device), train_d.to(device), train_e.to(device)]
train_a, train_b, train_c , train_d, train_e= torch.split(train_meta_data,int(len(train_meta_data)/5))
train_meta_data = [train_a.to(device), train_b.to(device), train_c.to(device), train_d.to(device), train_e.to(device)]
train_a, train_b, train_c , train_d, train_e= torch.split(train_mask,int(len(train_mask)/5))
train_mask = [train_a.to(device), train_b.to(device), train_c.to(device), train_d.to(device), train_e.to(device)]

# Displaying histogram
label_distribution = []
for label in LABELS:
    label_distribution.append(np.count_nonzero(df_train['label'] == label))

print(label_distribution)
plt.bar(LABELS, label_distribution)
# plt.show()
plt.close() # Close a figure window

# Encoding labels
label_encoder = LabelEncoder()
label_encoder.fit(LABELS)
joblib.dump(label_encoder, 'data/label_encoder.plk')

df_train['label'] = label_encoder.transform(df_train['label'])
train_labels = df_train.pop('label')
train_labels = torch.tensor(train_labels).to(device)

train_a, train_b, train_c , train_d, train_e= torch.split(train_labels,int(len(train_labels)/5))
train_labels = [train_a.to(device), train_b.to(device), train_c.to(device), train_d.to(device), train_e.to(device)]

df_eval['label'] = label_encoder.transform(df_eval['label'])
eval_labels = df_eval.pop('label')
eval_labels = torch.tensor(eval_labels).to(device)

df_test['label'] = label_encoder.transform(df_test['label'])
test_labels = df_test.pop('label')
test_labels = torch.tensor(test_labels).to(device)

print(df_train)
print(df_train.shape)

META_SIZE = len(train_meta_data[0])
META_HIDDEN_SIZE = 128
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(LABELS)
BATCH_SIZE = 8

model = FakeNewsClassifier(meta_size=META_SIZE, meta_hidden_size=META_HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)

# Moving tensors to device
eval_sentences = eval_sentences.to(device)
eval_meta_data = eval_meta_data.to(device)
eval_mask = eval_mask.to(device)

test_sentences = test_sentences.to(device)
test_meta_data = test_meta_data.to(device)
test_mask = test_mask.to(device)

# Trainloader
train_dataset = []
for i in range(len(train_mask)):
    train_dataset.append(TensorDataset(train_sentences[i], train_meta_data[i], train_mask[i], train_labels[i]))

# Valloader
eval_dataset = TensorDataset(eval_sentences, eval_meta_data, eval_mask, eval_labels)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Testloader
test_dataset = TensorDataset(test_sentences, test_meta_data, test_mask, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Hyperparameters
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

#creating graphs:
# x = np.linspace(0, NUM_EPOCHS-1, NUM_EPOCHS)
x = np.array(list(range(NUM_EPOCHS)))
y_loss = []
y_f1 = []
y_acc = []
y_val_loss = []
y_val_f1 = []
y_val_acc = []

# Instantiate MulticlassF1Score and MulticlassAccuracy
f1_score = MulticlassF1Score(num_classes=len(LABELS)).to(device)
accuracy = MulticlassAccuracy(num_classes=len(LABELS)).to(device)


# Tensorboard writer
#writer = SummaryWriter('runs/my_experiment')
starting_time = time.time()

prev_acc = 0.0
for epoch in range(NUM_EPOCHS):

    buffer_dataset = []
    k_fold = epoch % 5
    for i in range(len(train_dataset)):
        if i == k_fold:
            buffer_eval = train_dataset[i]
        else:
            buffer_dataset.append(train_dataset[i])

    buffer_dataset = ConcatDataset(buffer_dataset)
    train_loader = DataLoader(buffer_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(buffer_eval, batch_size=BATCH_SIZE, shuffle=True)

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    global_step = 0
    print('Epoch:', epoch+1)
    for num_batch, (inputs, metadatas, masks, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs, metadatas, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #writer.add_scalar('loss', loss.item(), global_step)
        total_loss += loss.item()
        total_f1 += f1_score(outputs, labels)
        total_acc += accuracy(outputs,labels)
        print(f'{num_batch}/{len(train_loader)}')
    average_loss = total_loss / len(train_loader)
    average_f1 = total_f1 / len(train_loader)
    average_acc = total_acc / len(train_loader)
    y_loss.append(average_loss)
    y_f1.append(average_f1.item())
    y_acc.append(average_acc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {average_loss} - F1: {average_f1} - Accuracy: {average_acc}")

    #validation set
    total_loss = 0.0
    total_f1 = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for num_batch, (inputs, metadatas, masks, labels) in enumerate(eval_loader):
            outputs = model(inputs, metadatas, masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_f1 += f1_score(outputs,labels)
            total_acc += accuracy(outputs,labels)
            print(f'{num_batch}/{len(eval_loader)}')
        average_loss = total_loss / len(eval_loader)
        average_f1 = total_f1 / len(eval_loader)
        average_val_acc = total_acc / len(eval_loader)
        y_val_loss.append(average_loss)
        y_val_f1.append(average_f1.item())
        y_val_acc.append(average_val_acc)
        print(f"Validation {epoch+1}/{NUM_EPOCHS} - Loss: {average_loss} - F1: {average_f1} - Accuracy: {average_val_acc}")
    
    if abs(average_acc-prev_acc)<0.03:
        break
    prev_acc = average_acc


print('Training time elapsed:', time.time() - starting_time)

plt.plot(x, y_loss, label="Training loss")
plt.plot(x, y_val_loss, label="Validation loss")
plt.legend()
plt.savefig("results_loss.png")
plt.close()

plt.plot(x, y_f1, label="Training F1")
plt.plot(x, y_val_f1, label="Validation F1")
plt.legend()
plt.savefig("results_F1.png")
plt.close()

plt.plot(x, y_acc, label="Training accuracy")
plt.plot(x, y_val_acc, label="Validation accuracy")
plt.legend()
plt.savefig("results_accuracy.png")
plt.close()

# Testing set
total_loss = 0.0
total_f1 = 0.0
total_acc = 0.0
with torch.no_grad():
    for num_batch, (inputs, metadatas, masks, labels) in enumerate(test_loader):
        outputs = model(inputs, metadatas, masks)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_f1 += f1_score(outputs,labels)
        total_acc += accuracy(outputs,labels)
        print(f'{num_batch}/{len(test_loader)}')
    average_f1 = total_f1 / len(test_loader)
    average_loss = total_loss / len(test_loader)
    average_acc = total_acc / len(test_loader)
    print(f"Testing {epoch+1}/{NUM_EPOCHS} - Loss: {average_loss} - F1: {average_f1} - Accuracy: {average_acc}")

#writer.close()

# Saving the PyTorch model
MODEL_PATH = 'models/model.pt'

# Classic model export
#torch.save(model, MODEL_PATH)
torch.save(model.state_dict(), MODEL_PATH)

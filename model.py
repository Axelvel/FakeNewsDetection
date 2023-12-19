import torch.nn as nn
from transformers import AutoModel


BERT_MODEL = "bert-base-uncased"

class FakeNewsClassifier(nn.Module):
    def __init__(self):
        super(FakeNewsClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(BERT_MODEL)

    def forward(self, x, meta):
        return x

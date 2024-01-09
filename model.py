import torch.nn as nn
from transformers import AutoModel


BERT_MODEL = "bert-base-uncased"

class FakeNewsClassifier(nn.Module):
    def __init__(self, meta_size, meta_hidden_size, hidden_size, output_size):
        super(FakeNewsClassifier, self).__init__()

        self.bert_hidden_size = 768
        self.dropout_rate = 0.5
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.meta_size = meta_size
        self.meta_hidden_size = meta_hidden_size

        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        self.metadata_fc = nn.Linear(self.meta_size, self.meta_hidden_size)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Original model
        """
        self.fc = nn.Sequential(
            nn.Linear(self.bert_hidden_size + self.meta_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.output_size)
        )
        """

    def forward(self, inputs, metadatas, attention_mask):
        bert_output = self.bert(inputs, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

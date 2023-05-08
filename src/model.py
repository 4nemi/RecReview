import config
from transformers import AutoModel, AutoConfig
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert_config = AutoConfig.from_pretrained(config.MODEL, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(config.MODEL, config=self.bert_config)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert_config.hidden_size, 1)

    def forward(self, ids, mask, token_type_ids):
        #最終層の[cls]トークンのみを取得する
        feature = self.bert(ids, 
                            attention_mask=mask,
                            token_type_ids=token_type_ids
                            )["last_hidden_state"][:, 0, :]
        output = self.bert_drop(feature)
        output = self.out(output)
        return output
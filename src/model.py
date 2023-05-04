import config
from transformers import AutoModel, AutoConfig
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert_config = AutoConfig.from_pretrained(config.MODEL)
        self.bert = AutoModel.from_pretrained(config.MODEL, config=self.bert_config)
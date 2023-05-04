import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, item):
        review = str(self.review[item])
        inputs = self.tokenizer.encode_plus(
            review, 
            None, 
            add_special_tokens=True, 
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        target = torch.tensor(self.target[item], dtype=torch.float)
        return inputs, target
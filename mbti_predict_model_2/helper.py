from torch.utils.data import Dataset
import torch
import re
import numpy as np


class Transformer_Dataset(Dataset):
    def __init__(self, input_data, output_labels, seq_len=512, device = "cuda"):
        
        self.seq_len = seq_len
        self.corpus_lines = len(output_labels)
        self.input_data = input_data
        self.output_labels = output_labels
        self.device = device
        
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        output = {key: val[item] for key, val in self.input_data.items()}
        output['labels'] = self.output_labels[item]

        return {key: torch.tensor(value).to(self.device) for key, value in output.items()}
    

def preprocess(dataset, tokenizer, seq_len = 512, label = False):
    
    def remove_urls(text):
        # Regex pattern to match URLs
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    if not label:
        for i, data in enumerate(dataset):
            dataset[i] = remove_urls(data)[:512]

    output = tokenizer(dataset, max_length = seq_len, padding='max_length')
    
    return output


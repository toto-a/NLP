import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, text_data :str, seq_length :int=25, block : int = 1):
        self.block=block
        self.chars = list(set(text_data))
        self.data_size, self.vocab_size = len(text_data), len(self.chars)

        self.char_to_idx = {ch:i for i,ch in enumerate(self.chars)}
        self.idx_to_char = {values:keys for keys, values in self.char_to_idx.items()}
        self.seq_length = seq_length
        self.X=self.string_to_vector(text_data)
    

    @property
    def X_string(self):
        return self.vector_to_string(self.X)

    def __len__(self):
        return len(self.X)//self.seq_length-1
    
    def __getitem__(self, index) :
        index=index+self.block
        X=torch.tensor(self.X[index*self.seq_length:(index+1)*self.seq_length]).float()
        y=torch.tensor(self.X[index*self.seq_length+1:(index+1)*self.seq_length+1]).float()
        return X, y
        
    

    def string_to_vector(self, name : str) :
        return [self.char_to_idx[ch] for ch in name]
    
    def vector_to_string(self, name : list):
        return "".join([self.idx_to_char[i] for i in name])

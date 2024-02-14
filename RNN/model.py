import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import label2lang,name2tensor,tensor2name


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hideen_size = hidden_size
        self.in2hid = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate=nn.Sigmoid()
        self.in2out = nn.Linear(input_size + hidden_size, output_size)
    
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.gate(self.in2hid(combined))
        output = self.in2out(combined)
        return output, hidden
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.zeros(1, self.hideen_size))
    
    @torch.no_grad()
    def predict(self,name):
        self.eval()
        hidden_state= self.init_hidden()
        for char in name:
            output, hidden_state = self(char, hidden_state)
        pred=torch.argmax(output,dim=1,keepdim=True).item()

        print(f' For name {tensor2name(name)} predicted language is {label2lang[pred]}')
        self.train()
        return label2lang[pred]
            
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import label2lang,name2tensor,tensor2name


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hideen_size = hidden_size
        self.in2hid = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.gate=nn.Tanh()
        self.in2out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        
        combined=self.in2hid(x)+self.h2h(hidden)
        hidden = self.gate(combined)
        output = self.in2out(combined)
        return output, hidden
    
    def init_hidden(self,batch_size=16):
        return nn.init.kaiming_uniform_(torch.zeros(batch_size, self.hideen_size))
    
    def predict(self,name):
        hidden_state= self.init_hidden()
        for char in name:
            output, hidden_state = self(char, hidden_state)
        pred=torch.argmax(output,dim=1,keepdim=True).item()

        # print(f' For name {tensor2name(name)} predicted language is {label2lang[pred]}')
        return label2lang[pred]
    
    def generate(self,data,pred_len=1000):
        pred=data.vector_to_string([torch.randint(0, data.vocab_size-1, (1,)).item() ])
        hidden_state= self.init_hidden()

        for i in range(pred_len-1):
            last_char=data.char_to_idx[pred[-1]]
            X,hidden_state=torch.tensor([last_char]).float(),hidden_state.float()
            output, hidden_state = self(X, hidden_state)
            prob=F.softmax(output,dim=-1)
            result=torch.multinomial(prob,1)
            result_list=result.view(-1).tolist()
            pred+=data.vector_to_string(result_list)
        
        return pred

            
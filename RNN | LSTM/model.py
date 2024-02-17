import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import label2lang,name2tensor,tensor2name


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,cell_type: str="RNN") -> None:
        super().__init__()
        self.type_=cell_type
        self.hideen_size = hidden_size
        self.in2hid = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.gate=nn.Tanh()
        self.in2out = nn.Linear(hidden_size, output_size)

        ### For LSTM
        self.forget_gate=nn.Sigmoid()
        self.input_gate=nn.Sigmoid()
        self.output_gate=nn.Sigmoid()
        self.contorl_gate=nn.Tanh()
    
    def forward(self, x, hidden,cell_state=None):
        
        combined=self.in2hid(x)+self.h2h(hidden)
        if self.type_=="RNN":
            hidden = self.gate(combined)
            output = self.in2out(combined)
        
        else :
            ## Bottom
            ft=self.forget_gate(combined)
            it=self.input_gate(combined)
            ct_tilde=self.contorl_gate(combined)
            ot=self.output_gate(combined)
            it_ct_tilde=it+ct_tilde

            ## Top
            ct=cell_state
            ct_0=ft*ct
            ct_1=it_ct_tilde+ct_0
           
            ht=ot*F.tanh(ct_1)

            hidden=ht
            output=self.in2out(ct_1)

        return output, hidden
    
    def init_hidden(self,batch_size=16):
        return nn.init.kaiming_uniform_(torch.zeros(batch_size, self.hideen_size))
    
    def init_cell_state(self,batch_size=16):
        return nn.init.kaiming_uniform_(torch.zeros(batch_size, self.hideen_size))

    def predict(self,name):
        hidden_state= self.init_hidden()
        for char in name:
            output, hidden_state = self(char, hidden_state)
        pred=torch.argmax(output,dim=1,keepdim=True).item()

        # print(f' For name {tensor2name(name)} predicted language is {label2lang[pred]}')
        return label2lang[pred]
    
    def generate(self,data,pred_len=1000,cell_state=None):
        pred=data.vector_to_string([torch.randint(0, data.vocab_size-1, (1,)).item() ])
        hidden_state= self.init_hidden()
        if cell_state is None and self.type_=="LSTM":
            cell_state=self.init_cell_state().float()

        for i in range(pred_len-1):
            last_char=data.char_to_idx[pred[-1]]
            X,hidden_state=torch.tensor([last_char]).float(),hidden_state.float()
            if self.type_!="RNN":
                output, hidden_state = self(X, hidden_state,cell_state)
            else :
                output, hidden_state = self(X, hidden_state) 
            prob=F.softmax(output,dim=-1)
            result=torch.multinomial(prob,1)
            result_list=result.view(-1).tolist()
            pred+=data.vector_to_string(result_list)
        
        return pred

            
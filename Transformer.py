import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math 

batch_size=32
block_size=8
max_iters=300
eval_interval=300
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
dropout=0.2



class InputEmbedding(nn.Module):
    def __init__(self,d_model, vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    


class PositionalEmbedding(nn.Module) :
    def __init__(self,d_model,seq_len) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(-torch.arange(0,d_model,2)*math.log(10000)/d_model)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(dim=0) #batch dimension
        self.register_buffer("pe",pe)

    def forward(self,x):
        out=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        out=self.dropout(out)

        return out
        


class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff) -> None:
        #d_ff size of the hidden state
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )
    def forward(self, x):
        #(Batch, seq_len,d_model)-> (Batch,seq_len,d_ff) -> (Batch_size,seq_len,d_model)
        
        out=self.net(x)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_head, head_size,masked=None) -> None:
        super().__init__()

        self.mask=masked
        self.h=num_head
        self.d_model=d_model
        self.dh=head_size

        self.q=nn.Linear(d_model,head_size)
        self.k=nn.Linear(d_model,head_size)
        self.v=nn.Linear(d_model,head_size)         
        self.dropout=nn.Dropout(dropout)

        if not masked : 
            self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
       
        self.proj=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    
    @staticmethod
    def attention(query,key,value,mask,dropout : nn.Dropout):
        dhead=query.shape[-1]
        

        attention_scores=query@key.transpose(-2,-1)*dhead**-0.5
        if  mask is not None: 
            attention_scores=attention_scores.masked_fill(mask==0,value=float('-inf'))    

        attention=F.softmax(attention_scores,dim=-1)
        if dropout is not None : 
            attention_scores=dropout(attention_scores)

        attention=attention_scores@value
        return attention,attention_scores

        
    
    def forward(self,query,key,value,mask):

      
        q=self.q(query)
        k=self.k(key) 
        v=self.v(value)

        B,seq_len,embd=q.shape

        # (Batch,seq_len,d_model)->(Batch,seq_len,n_head,head_size)->(Batch,n_head,seq_len,head_size)
        q=q.view(-1,self.h,self.dh).transpose(1,2)
        k=k.view(-1,self.h,self.dh).transpose(1,2)
        v=v.view(-1,self.h,self.dh).transpose(1,2)

        x,attention_score=MultiHeadAttention.attention(q,k,v,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(-1,self.h*self.dh)#(Batch,n_head,seq_len,head_size)->(Batch,seq_len,n_head,head_size) ->(Batch,seq_len,d_model)

        out=self.proj(x)

    # (Batch,seq_len,d_model)->(Batch,seq_len,d_model)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        
        assert d_model % n_head ==0
        
        self.sa_heads=MultiHeadAttention(d_model=d_model,num_head=n_head,head_size=d_model//n_head)
        self.mlp=FeedForward(d_model)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)
    
    def forward(self,x):
        out=self.sa_heads(self.ln1(x))
        out=out + x
        out=self.ln2(self.mlp(out)) + out

        return out


class Encoder(nn.Module) :
    def __init__(self,n_layers,d_model,n_head) -> None:
        super().__init__()
        self.encoder_block=nn.Sequential(*[EncoderBlock(d_model=d_model,n_head=n_head) for _ in range(n_layers)])
        self.ln=nn.LayerNorm(d_model)
    

    def forward(self,x):
        out=self.encoder_block(x)
        out=self.ln(out)

        return out





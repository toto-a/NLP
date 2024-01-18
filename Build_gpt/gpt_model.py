import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import math
from dataclasses import dataclass
from typing import Optional,Tuple


@dataclass
class GPTModelArgs :
    dim :int=4096
    n_layers : int =32
    n_heads: int=32 ##Number of heads for the queries,key and values
    vocab_size: int = -1 ##Tokenizer param
    dim_ffn_multiple : Optional[int]=None
    norm_eps : float =None
    dropout : Optional[float] =None
    device: str=None
    max_batch_size: int =32
    seq_len: int = 512


class PositionalEmbedding(nn.Module):
    def __init__(self, args :GPTModelArgs) -> None:
        super().__init__()
        self.seq_len=args.seq_len
        self.dim=args.dim

        pos=torch.arange(0,self.seq_len,dtype=torch.float)
        div=torch.exp(-torch.arange(0,self.seq_len,2)*math.log(10000)/self.dim)
        PE=torch.zeros((self.seq_len,self.dim))

        PE[:,::2]=torch.sin(pos*div)
        PE[:,1::2]=torch.cos(pos*div)

        PE=PE.unsqueeze(0) ##Batch dimension
        self.register_buffer('pe',PE)
    

    def forward(self, x:torch.Tensor) :
        T=x.shape[1] ##seq_len
        out=(x+self.pe[:,:T,:]).requires_grad_(False) ##Absolute position
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, args: GPTModelArgs) -> None:
        super().__init__()
        self.dim=args.dim
        self.n_head=args.n_heads

        self.wq=nn.Linear(self.dim,self.dim)
        self.wk=nn.Linear(self.dim,self.dim)
        self.wv=nn.Linear(self.dim,self.dim)
        self.wo=nn.Linear(self.dim,self.dim)
        self.dropout=nn.Dropout(args.dropout) if args.dropout is not None else nn.Dropout(0.2)
        ###Causal Mask
        self.register_buffer('tril',torch.tril(torch.ones(args.seq_len,args.seq_len).view(1,1,args.seq_len,args.seq_len)))
    
  

    def forward(self,x: torch.tensor):

        B,T,C=x.shape
        q=self.wq(x)
        k=self.wk(x)
        v=self.wv(x) ##(B,T,dim:embedding)

        ##(B,nh,T,head_size)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)


        ##Causal self_attention
        c_att=(q@k.transpose(-1,-2))*(1/math.sqrt(k.size(-1))) ##Divide by head size to have unit gaussian
        c_att=c_att.masked_fill(self.tril[:,:,:T,:],float('-inf'))
        c_att=F.softmax(att)


        att=c_att@v
        y=att.tranpose(1,2).contiguous().view(B,T,C)

        y=self.dropout(att)
        return y


           


        










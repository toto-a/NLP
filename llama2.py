import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from utils_llama import SwiGlu



dropout=0.2
device='cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ModelArgs :
    dim :int=4096
    n_layers : int =32
    n_heads: int=32 ##Number of heads for the queries
    n_kv_heads : Optional[int]=None ##Number of heads for the keys and value
    vocab_size: int = -1 ##Tokenizer param
    mutiple_of: int=256
    dim_ffn_multiple : Optional[float]=None
    norm_dps : float =None


    ##KV cache
    max_batch_size: int =32
    max_seq_len: int =2048

    device: str=None




class InputEmbedding(nn.Module) :
    def __init__(self, vocab_size,d_model) -> None:
        super().__init__()
        self.d_model=d_model
        self.token_embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x) :
        out=self.token_embedding(x)*self.d_model**-0.5
        return out


class RoPe(nn.Module) :
    def __init__(self, base,seq_len,d_model) -> None:
        super().__init__()
        self.base=base
        self.d=d_model
        self.cos_cached=None
        self.sin_cached=None
        
    def _build_cache(self,x):
        ##Return a value if already cached
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0] :
            return
        

        B,seq_len,d_model=x.shape
        theta=1./(10000**(torch.arange(0,self.d,2)/self.d)).float().to(device)
        seq_indx=torch.arange(0,seq_len,device=device).float().to(device)

        ##Product of index and theta (m*theta)
        idx_theta=torch.einsum('n,d->nd',seq_indx,theta)

        ##..And repeat(2 time)
        idx_theta2=torch.cat([idx_theta,idx_theta],dim=1)


        self.cos_cached=idx_theta2.cos()[:,None,None,:]
        self.sin_cached=idx_theta2.sin()[:,None,None,:]




    def forward(self,x) :
        ## x is the input of a query or a key , of shape (batch,seq_len,n_heads,d_head)
        self._build_cache(x)

        return 




class RMS(nn.Module) :
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x) :
        return


class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff) -> None:
        #d_ff size of the hidden state
        super().__init__()
        self.swish_glu=SwiGlu(d_ff,d_ff)

        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff),
            self.swish_glu,
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )
    def forward(self, x):
        #(Batch, seq_len,d_model)-> (Batch,seq_len,d_ff) -> (Batch_size,seq_len,d_model)
        
        out=self.net(x)
        return out



class RoPeMultiHeadAttention(nn.Module):
    ##Head of different size for the query, (key and value)
    ##RoPe used only for the query and the key
    def __init__(self, d_model,num_head, head_size_query,head_size) -> None:
        super().__init__()

        self.h=num_head
        self.d_model=d_model
        self.dkv=head_size
        self.dq=head_size_query

        self.q=nn.Linear(d_model,d_model)
        self.kv=nn.Linear(d_model*d_model,d_model)
        self.dropout=nn.Dropout(dropout)
       
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
        k,v=torch.chunk(self.kv(torch.cat([k,v],dim=1)),2,dim=-1)

        B,seq_len,embd=q.shape

        # (Batch,seq_len,d_model)->(Batch,seq_len,n_head,head_size)->(Batch,n_head,seq_len,head_size)
        q=q.view(B,seq_len,self.h,self.dh).transpose(1,2)
        k=k.view(B,seq_len,self.h,self.dh).transpose(1,2)
        v=v.view(B,seq_len,self.h,self.dh).transpose(1,2)

        x,attention_score=RoPeMultiHeadAttention.attention(q,k,v,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(B,seq_len,self.h*self.dh)#(Batch,n_head,seq_len,head_size)->(Batch,seq_len,n_head,head_size) ->(Batch,seq_len,d_model)

        out=self.proj(x)

            # (Batch,seq_len,d_model)->(Batch,seq_len,d_model)
        return out



class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size) -> None:
        super().__init__()
        self.norm=RMS()
        self.proj=nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(self.norm(x),dim=-1,dtype=torch.int64))


class Decoder(nn.Module):
    def __init__(self,d_model, n_head,d_ff) :
        self.sa_heads=RoPeMultiHeadAttention(d_model,n_head,d_model//n_head)
        self.ffn=FeedForward(d_model,d_ff)
        self.rms=RMS(d_model)
    
    def forward(self,x):
        _x=x
        out=self.sa_heads(x)
        out=_x+self.rms(out)

        return out




class Llama2(nn.Module) :
    def __init__(self,args:ModelArgs) -> None:
        super().__init__()
        self.token_embedding=InputEmbedding(args.vocab_size,args.dim)
        self.in_norm=RMS()
        self.out_norm=RMS()
        self.ffn=FeedForward(args.dim, args.dim_ffn_multiple)


    def forward(self,x) :
        x=self.token_embedding(x)


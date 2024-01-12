import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from utils_llama import SwiGlu,RMSnorm,FeedForward,repeat_kv



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
    norm_eps : float =None


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
    def __init__(self, head_dim) -> None:
        super().__init__()
        self.d=head_dim
        self.cos_cached=None
        self.sin_cached=None
        
    def _build_cache(self,x):
        ##Return a value if already cached
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0] :
            return
        

        B,seq_len,d_model=x.shape
        theta=1./(10000**(torch.arange(0,self.d,2)/self.d)).float().to(device)
        seq_indx=torch.arange(0,seq_len,device=device).float().to(device)

        ##Product of index and theta (m*theta) (Seq_len, head_dim/2)
        idx_theta=torch.einsum('n,d->nd',seq_indx,theta)

        ##..And repeat(2 time) (Seq_len,head_dim)
        # idx_theta2=torch.cat([idx_theta,idx_theta],dim=1)
        # self.cos_cached=idx_theta2.cos()[None,:,None,:]
        # self.sin_cached=idx_theta2.sin()[None,:,None,:]

        idx_theta_complex=torch.polar(torch.ones(idx_theta),idx_theta)
        return idx_theta_complex

    def forward(self,x :torch.Tensor) :
        freq_complex=self._build_cache(x) ##(Seq_len,head_dim/2)
        freq_complex=freq_complex[None,:,None,:] ##(1,Seq_len,1,head_dim/2)

        ## x->(B,seq_len,H,head_dim)->(B,seq_len,H,head_dim/2)
        x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,-2)) ##Reshape x into [xd_1,xd] into the complex plane

        ## x_rope_complex->(B,seq_len,H,head_dim/2) .. beacause of broadcast
        x_rope_complex=x_complex * freq_complex

        ## x_rope->(B,seq_len,H,head_dim/2, 2), the [a+ib] become [a b]
        x_rope=torch.view_as_real(x_rope_complex)
        
        ##x_rope=x_rope.flatten(start_dim=-2, end_dim=-1)
        
        ## x_rope-> (B,seq_len,H,head_dim)
        x_rope=x_rope.reshape(*x.shape)

        return x_rope.type_as(x).to(device)





class GPASelfAttention(nn.Module):
    ##Head of different size for the query, (key and value)
    ##RoPe used only for the query and the key
    ##KV cache
    def __init__(self, args :ModelArgs) -> None:
        super().__init__()

        self.h_query=args.n_heads ##Head for the query
        self.h_kv=args.n_heads if args.n_kv_heads is  None else args.n_kv_heads ##Heads for the keys and values 
        self.d_model=args.dim

        self.n_rep=self.h_query//self.h_kv ##Indicate hwo much the heads of K and V be repeated to match the query
        self.dkv=self.d_model//self.h_kv ##Head dimension
        self.dq=self.d_model//self.h_query

        self.wq=nn.Linear(args.dim,args.n_heads*self.dq)
        self.wkv=nn.Linear(args.dim*args.dim,args.n_kv_heads*self.dkv)
        self.proj=nn.Linear(args.n_heads*self.dq,args.dim)
        self.dropout=nn.Dropout(dropout)

        self.rope_q=RoPe(self.dq)
        self.rope_k=RoPe(self.dkv)
        self.cache_k=torch.zeros((args.max_batch_size,args.max_seq_len,self.h_kv,self.dq))
        self.cache_v=torch.zeros((args.max_batch_size,args.max_seq_len,self.h_kv,self.dq))

    
    @staticmethod
    def attention(query,key,value,dropout : nn.Dropout):
        dhead=query.shape[-1]
        

        attention_scores=query@key.transpose(-2,-1)*dhead**-0.5
        attention=F.softmax(attention_scores,dim=-1)
        if dropout is not None : 
            attention_scores=dropout(attention_scores)

        attention=attention_scores@value
        return attention,attention_scores

        
    
    def forward(self,x,start_pos):

        B,seq_len,dim=x.shape
      
        q=self.wq(x) ##->(B,seq_len,H_Q*Head_dim)
        k,v=torch.chunk(self.wkv(torch.cat([k,v],dim=1)),2,dim=-1)


        # (Batch,seq_len,d_model)->(Batch,seq_len,n_head,head_size)
        q=q.view(B,seq_len,self.h_query,self.dq)
        k=k.view(B,seq_len,self.h_kv,self.dkv)
        v=v.view(B,seq_len,self.h_kv,self.dkv)

        #RoPE does not change shape
        xq=self.rope_q(q)
        xk=self.rope_k(k)

        ##Replace the cache for this value
        self.cache_k[:B,start_pos:start_pos+seq_len,:]=xk
        self.cache_v[:B,start_pos:start_pos+seq_len,:]=v

        ##Retrieve cached keys and values to compute self_attention
        #(B,seq_lenKV, H_KV, D_KV)
        keys=self.cache_k[:B,0:start_pos+seq_len]
        values=self.cache_v[:B,0:start_pos+seq_len]

        ##Repeat heads of the cached values to reach the numbers of head for the query
        keys=repeat_kv(keys,self.n_rep)
        values=repeat_kv(keys,self.n_rep)

        ##Reshape to see n_heads as a batch dim
        xq=xq.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)


        x,attention_score=GPASelfAttention.attention(q,k,v,self.dropout)
        #(Batch,n_head,seq_len,head_size)->(Batch,seq_len,n_head,head_size) ->(Batch,seq_len,d_model)
        x=x.transpose(1,2).contiguous().view(B,seq_len,self.h_query*self.dq)

        out=self.proj(x)

        return out



class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size) -> None:
        super().__init__()
        self.norm=RMSnorm(d_model)
        self.proj=nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(self.norm(x),dim=-1,dtype=torch.int64))


class Encoder(nn.Module):
    def __init__(self,args:ModelArgs) :
        super().__init__()

        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=self.dim//self.n_heads

        self.sa_heads=GPASelfAttention(args)
        self.ffn=FeedForward(self.dim,args.mutiple_of,args.dim_ffn_multiple)
        self.rms_bf_att=RMSnorm(self.dim,args.norm_eps)
        self.rms_ffn_norm=RMSnorm(self.dim,args.norm_eps)
    
    def forward(self,x,start_pos):

        out=x+self.sa_heads(self.rms_bf_att(x),start_pos)
        out=out+self.ffn(self.rms_ffn_norm(out))


        return out




class Llama2(nn.Module) :
    def __init__(self,args:ModelArgs) -> None:
        super().__init__()
        self.token_embedding=InputEmbedding(args.vocab_size,args.dim)
        self.encoder_blocks=nn.ModuleList([Encoder(args) for _ in range(args.n_layers)])
        self.proj=ProjectionLayer(args.dim,args.vocab_size)


    def forward(self,x) :
        x=self.token_embedding(x)

        for layer in self.encoder_blocks :
            x=layer(x)
        
        x=self.proj(x)

        return x


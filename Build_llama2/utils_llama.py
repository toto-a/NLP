import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass



class SwiGlu(nn.Module):
    def __init__(self,beta=1) -> None:
        super().__init__()
        self.beta=beta

    def forward(self,x) :
        out=x*F.sigmoid(self.beta*x)
        return out

class RMSnorm(nn.Module):
    def __init__(self,dim,eps=1e-6) -> None:
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dim),requires_grad=True)
        self.eps=eps
        
    def _norm(self,x):
        return x*(torch.rsqrt((x.pow(2)).mean(-1,keepdim=True)+self.eps)) 


    def forward(self,x) :
        
        return  self.weight*self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
        def __init__(self,d_model,multiple_of,ffn_dim_multiplier) :
            super().__init__()
            hidden_dim = 4 * d_model
            hidden_dim = int(2 * hidden_dim / 3)

            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier* hidden_dim)

            # Round the hidden_dim to the nearest multiple of the multiple_of parameter
            hidden_dim = multiple_of * ((hidden_dim +multiple_of- 1) // multiple_of)

            self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
            self.swiglu=SwiGlu()
        
        def forward(self,x) :
            # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
            swish = self.swiglu(self.w1(x))
            # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
            x_V = self.w2(x)

            # (m, seq_len, hidden_dim)
            x = swish * x_V

            # (m, seq_len, hidden_dim) --> (m, seq_len, dim)
            return self.w3(x)




def repeat_kv(x,n_rep) :
    B,seq_len,n_kv,d_kv=x.shape
    if n_rep==1 :
        return x
    else :
        #(B,seq_len,n_kv_heads,1,d_kv) ->#(B,seq_len,n_kv_heads,n_rep,d_kv) -> (B,seq_len,n_kv_heads*n_rep,d_kv)
        x=x[:,:,:,None,:].expand(B,seq_len,n_kv,n_rep,d_kv).reshape(B,seq_len,n_kv*n_rep,d_kv)
        return x


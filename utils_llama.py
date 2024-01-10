import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass



class SwiGlu(nn.Module):
    def __init__(self,d_model,d_f,d_ff) -> None:
        super().__init__()
        self.L1=nn.Linear(d_model,d_f)
        self.L2=nn.Linear(d_model,d_ff)


    def forward(self,x) :
        out=F.sigmoid(self.L1(x))@self.L2(x)
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

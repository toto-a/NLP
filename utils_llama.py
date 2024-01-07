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


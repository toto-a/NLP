import torch
import torch.nn as nn     
import torch.nn.functional as F   

from transformer_components import CausalAttention,MLP,RMSNorm




class Block(nn.Module) :
    def __init__(self, config):
        super().__init__()

        self.attention = CausalAttention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out
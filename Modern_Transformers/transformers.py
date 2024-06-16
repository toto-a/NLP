import torch
import torch.nn as nn     
import torch.nn.functional as F   
from dataclasses import dataclass

from transformer_components import CausalAttention,MLP,RMSNorm,CausalAttentionKVCache,PositionEmbeddings

@dataclass
class TransformerConfig :
    seq_len : int =512
    n_heads : int =12
    hidden_size : int = 768 
    k_hidden :int =4
    eps : float = 1e-12


    ##Flag
    return_pe=True



config=TransformerConfig()


pos={'sinusoid ' : PositionEmbeddings(config)}
att={"self_attention": CausalAttention(config), "kv_attention":CausalAttentionKVCache(config)}
norm={"rms":RMSNorm(config),"ln":nn.LayerNorm(config.hidden_size, config.eps)}




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




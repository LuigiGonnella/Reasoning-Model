from GLUFeedForwardNeuralNetwork.main import FeedForward
from GroupedQueryAttention.main import GroupedQueryAttention
from RMSNorm.main import RMSNorm
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
        d_in=cfg["emb_dim"],
        num_heads=cfg["n_heads"],
        head_dim=cfg["head_dim"],
        num_kv_groups=cfg["n_kv_groups"],
        qk_norm=cfg["qk_norm"],
        dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x, mask, cos, sin, start_pos=start_pos,cache=cache
            ) 
        
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache
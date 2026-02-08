import torch
from GLUFeedForwardNeuralNetwork.main import FeedForward
from GroupedQueryAttention.main import GroupedQueryAttention
from RMSNorm.main import RMSNorm
import torch.nn as nn
from QWEN3TransformerBlock.main import TransformerBlock
from RoPE.main import compute_rope_params

cfg = {
    "vocab_size": 151_936, # Vocabulary size
    "context_length": 40_960, # Length originally used during training
    "emb_dim": 1024, # Embedding dimension
    "n_heads": 16, # Number of attention heads
    "n_layers": 28, # Number of layers
    "hidden_dim": 3072, # Size of intermediate dim in FeedForward
    "head_dim": 128, # Size of the heads in GQA
    "qk_norm": True, # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8, # Key-Value groups for GQA
    "rope_base": 1_000_000.0, # The base in RoPE's "theta"
    "dtype": torch.bfloat16, # Lower-precision dtype to reduce memory
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"],
        dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList(
        [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"],
        bias=False, dtype=cfg["dtype"]
        )
        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]

        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0 # Track current position in KV cache
        
    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(      
                torch.ones(
                pos_end, pos_end, device=x.device, dtype=torch.bool
                ),
                diagonal=1
                )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0 # Not strictly necessary but helps torch.compile
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device,
                dtype=torch.bool),
                diagonal=1
            )
        mask = mask[None, None, :, :] #A Prefill: Shape (1, 1, T, T) to broadcast across batch and heads. Cached: Shape (1, 1, T, K+T) where T=new tokens, K=cached keys.

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                start_pos=pos_start,
                cache=blk_cache)

            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
    
    def reset_kv_cache(self):
        self.current_pos = 0

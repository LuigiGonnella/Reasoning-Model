import torch.nn as nn
from RMSNorm.main import RMSNorm
from RoPE.main import apply_rope
import torch

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None,
    qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(
        d_in, self.d_out, bias=False, dtype=dtype
        ) #d_out = num_heads * head_dim

        self.W_key = nn.Linear(
        d_in, num_kv_groups * head_dim, bias=False,dtype=dtype
        ) #d_out = num_kv_groups * head_dim
        self.W_value = nn.Linear(
        d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        ) #d_out = num_kv_groups * head_dim
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
    
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x) 
        keys = self.W_key(x) 
        values = self.W_value(x) 

        queries = queries.view(b, num_tokens, self.num_heads,
        self.head_dim).transpose(1, 2)

        keys_new = keys.view(b, num_tokens, self.num_kv_groups,
        self.head_dim).transpose(1, 2)

        values_new = values.view(b, num_tokens, self.num_kv_groups,
        self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)
        
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0 
            keys, values = keys_new, values_new

        next_cache = (keys, values)

        #queries.shape = (Batch, num_heads, Seq_Len, Head_Dim)
        #es. 8 heads

        #keys.shape = (Batch, num_kv_groups, Seq_Len, Head_Dim)
        #es. 2 groups

        keys = keys.repeat_interleave( 
        self.group_size, dim=1 
        ) #take the two heads of keys and repeat them until they become 8

        #repeat_interleave physically copies the keys, wasting memory
        #in optimized CUDA kernels this operation is implemented through 'broadcasting'
        #--> the same memory location is read multiple times without any wasting
        
        values = values.repeat_interleave( 
        self.group_size, dim=1 
        ) 

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / self.head_dim**0.5, dim=-1
        )

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache
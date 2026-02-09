import torch

#1) REGISTERING THE CACHE BUFFERS IN ATTENTION INIT
def __init__(self):
    #....
    self.register_buffer('cache_k', None, persistent=False)
    self.register_buffer('cache_v', None, persistent=False)
    self.ptr_current_pos = 0 #tracks the number of token to generate (with KV cache we generate just 1 token at the time)

#2) FORWARD PASS ATTENTION WITH CACHE
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...
    if use_cache:
        if self.cache_k is None: #FIRST PASS
            self.cache_k, self.cache_v = keys_new, values_new 
        else: #FOLLOWING PASSES
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1) #ADD TO CACHE
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
    
    #...
    num_tokens_Q = queries.shape[-2] #number of queries (rows of attention matrix)
    num_tokens_K = keys.shape[-2] #number of keys (and values, columns of attention matrix) 
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, #select the last ROW of the matrix, to generate next token
            #in the first forward pass ("Prefill"), when we have the whole user prompt, this number is equal to the number of tokens in the prompt, so that we can process the N tokens and generate the next token (the previous predicitons are trashed, we calculate them just to store K and V)
            #then, in the next passes ("Decoding"), this number will be 1, just generating one token at the time, remembering the past K and V
            :num_tokens_K #mask the future keys, that don't exist anyway with the KV cache (in Decoding, while in Prefill it is needed), contrarly to the non-KV cache case
        ]
        self.ptr_current_pos += num_tokens_Q #update current token generated

#3) ATTENTION RESET CACHE
def reset_cache(self): #clean cache between two different text generation calls
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0

#4) MODEL FORWARD CACHE
def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        # NEW

        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len #TRACKS the number of generated token so that the model can generate only next tokens without overlapping earlier ones
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # NEW
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

#5) MODEL RESET CACHE (for convenience)
def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    
#6) USING CACHE IN GEENRATION
def generate_text_simple_cached(
        model, idx, max_new_tokens, use_cache=True
    ):
    model.eval()

    ctx_len = model.pos_emb.num_embeddings  # max sup. len., e.g. 1024
    if use_cache:
        # Init cache with full prompt
        model.reset_kv_cache()
        with torch.no_grad():
            logits = model(idx[:, -ctx_len:], use_cache=True)

        for _ in range(max_new_tokens):
            # a) pick the token with the highest log-probability 
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            # b) append it to the running sequence
            idx = torch.cat([idx, next_idx], dim=1)
            # c) feed model only the new token
            with torch.no_grad():
                logits = model(next_idx, use_cache=True)
    else:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(idx[:, -ctx_len:], use_cache=False)
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)

    return idx
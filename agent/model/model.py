import torch
import torch.nn as nn
import math

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta **(torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def apply_rotary_embedding(xq, xk, freqs_cis):
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    xq_out = torch.view_as_real(xq_out).flatten(-2)
    xk_out = torch.view_as_real(xk_out).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask, freqs_cis, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k = apply_rotary_embedding(q, k, freqs_cis)

        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out), new_cache

class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, freqs_cis, cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), mask, freqs_cis, cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x, new_cache

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.embed = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        mask = torch.triu(
            torch.ones(block_size, block_size),
            diagonal=1
        ).bool()
        self.register_buffer("causal_mask", mask)

        head_dim = d_model // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, block_size)
        self.register_buffer("freqs_cis", freqs_cis)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, cache=None):
        B, T = x.shape
        x = self.embed(x)

        if cache is not None and cache[0] is not None:
            prev_len = cache[0][0].shape[2]
            start_pos = prev_len
        else:
            start_pos = 0

        freqs_cis = self.freqs_cis[start_pos:start_pos + T]

        total_len = start_pos + T

        mask = torch.triu(
            torch.ones(T, total_len, device=x.device),
            diagonal=start_pos + 1
        ).bool()

        new_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = block(x, mask, freqs_cis, layer_cache)
            new_cache.append(layer_new_cache)

        x = self.ln_f(x)
        logits =  self.lm_head(x)
        return logits, new_cache

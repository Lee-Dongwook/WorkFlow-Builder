import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(mask == 1, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, block_size=64):
        super().__init__()
        self.block_size = block_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.lm_head.weight = self.embed.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.embed(x) + self.pos(pos)

        mask = torch.triu(torch.ones(T, T), diagonal=1).to(x.device)

        for block in self.blocks:
            x = block(x, mask)

        return self.lm_head(x)

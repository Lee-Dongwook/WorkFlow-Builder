import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, block_size=32):
        super().__init__()
        self.block_size = block_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)

        x = self.embed(x) + self.pos_embed(pos)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        for block in self.blocks:
            x = block(x, src_mask=mask)
            
        return self.lm_head(x)

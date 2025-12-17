import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

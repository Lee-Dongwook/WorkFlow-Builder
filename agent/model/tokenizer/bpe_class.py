import json
import os
from .bpe import train_bpe, build_vocab, bpe_encode, decode

class BPETokenizer:
    def __init__(self, text=None, num_merges=500):
        if text is not None:
            self.merges = train_bpe(text, num_merges)
            self.stoi, self.itos = build_vocab(self.merges)
        else:
            self.merges = []
            self.stoi = {}
            self.itos = {}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        tokens = bpe_encode(text, self.merges)
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, ids):
        tokens = [self.itos[i] for i in ids]
        return decode(tokens)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "merges.json"), "w") as f:
            json.dump(self.merges, f)
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(os.path.join(path, "merges.json"), "r") as f:
            tokenizer.merges = [tuple(m) for m in json.load(f)]
        with open(os.path.join(path, "vocab.json"), "r") as f:
            tokenizer.stoi = json.load(f)
        tokenizer.itos = {int(i): s for s, i in tokenizer.stoi.items()}
        return tokenizer

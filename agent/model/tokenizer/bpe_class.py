import json
import os
from .bpe import train_bpe, build_vocab, bpe_encode, decode

class BPETokenizer:
    SPECIAL_TOKENS = {
        "<|bos|>": 0,
        "<|eos|>": 1,
        "<|pad|>": 2,
    }

    def __init__(self, text=None, num_merges=500):
        if text is not None:
            self.merges = train_bpe(text, num_merges)
            base_stoi, base_itos = build_vocab(self.merges)

            self.stoi = {**self.SPECIAL_TOKENS}
            self.itos = {v: k for k, v in self.SPECIAL_TOKENS.items()}

            offset = len(self.SPECIAL_TOKENS)
            for token, idx in base_stoi.items():
                self.stoi[token] = idx + offset
                self.itos[idx + offset] = token
        else:
            self.merges = []
            self.stoi = {**self.SPECIAL_TOKENS}
            self.itos = {v:k for k, v in self.SPECIAL_TOKENS.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def bos_id(self):
        return self.SPECIAL_TOKENS["<|bos|>"]

    @property
    def eos_id(self):
        return self.SPECIAL_TOKENS["<|eos|>"]

    @property
    def pad_id(self):
        return self.SPECIAL_TOKENS["<|pad|>"]

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = bpe_encode(text, self.merges)
        ids =  [self.stoi[t] for t in tokens if t in self.stoi]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
        

    def decode(self, ids, skip_special=True):
        if skip_special:
            ids = [i for i in ids if i not in self.SPECIAL_TOKENS.values()]
        tokens = [self.itos[i] for i in ids if i in self.itos]
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

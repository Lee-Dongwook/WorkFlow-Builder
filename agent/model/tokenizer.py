class CharTokenizer:
    def __init__(self, text):
        self.special = ["<PAD>", "<UNK>"]
        chars = sorted(list(set(text)))
        self.vocab = self.special + chars

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, s):
        return [self.stoi.get(c, self.stoi["<UNK>"]) for c in s]

    def decode(self, ids):
        pad = self.stoi["<PAD>"]
        return "".join(self.itos[i] for i in ids if i != pad)

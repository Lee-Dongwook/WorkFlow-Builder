import re
from collections import Counter, defaultdict

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

def byte_encode(text):
    return ''.join(
        BYTE_ENCODER[b]
        for b in text.encode("utf-8")
    )

def get_pairs(word):
    return set(zip(word[:-1], word[1:]))

def train_bpe(text, vocab_size=8000):
    text = byte_encode(text)
    words = text.split()

    vocab = Counter(tuple(word) for word in words)
    merges = []

    while len(vocab) < vocab_size:
        pairs = Counter()

        for word, freq in vocab.items():
            for pair in get_pairs(word):
                pairs[pair] += freq

        if not pairs:
            break

        best = max(pairs, key=pairs.get)
        merges.append(best)

        new_vocab = Counter()
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] += freq

        vocab = new_vocab

    return merges

def bpe_encode(text, merges):
    text = byte_encode(text)
    tokens = list(text)

    for a, b in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(a+b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens

def build_vocab(merges):
    vocab = set()
    for a, b in merges:
        vocab.add(a)
        vocab.add(b)
        vocab.add(a+b)

    vocab = sorted(vocab)
    stoi = {s:i for i,s in enumerate(vocab)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def decode(tokens):
    text = ''.join(tokens)
    byte_arr = bytearray([BYTE_DECODER[c] for c in text])
    return byte_arr.decode("utf-8", errors="replace")

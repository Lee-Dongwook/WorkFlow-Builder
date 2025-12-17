import torch
from model import TinyGPT
from tokenizer import CharTokenizer

block_size = 32

text = open("data/train.txt").read()
tokenizer = CharTokenizer(text)

model = TinyGPT(tokenizer.vocab_size)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def generate(start, length=100):
    ids = tokenizer.encode(start)
    for _ in range(length):
        x = torch.tensor([ids[-block_size:]])
        logits = model(x)
        next_id = torch.argmax(logits[0, -1]).item()
        ids.append(next_id)
    return tokenizer.decode(ids)

print(generate("h"))

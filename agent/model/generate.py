import torch
from model import TinyGPT
from tokenizer import CharTokenizer

text = open("data/train.txt").read()
tokenizer = CharTokenizer(text)

model = TinyGPT(tokenizer.vocab_size)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def generate(start, length=100):
    ids = tokenizer.encode(start)
    for _ in range(length):
        x = torch.tensor([ids[-1]])
        logits = model(x)
        next_id = torch.argmax(logits, dim=-1).item()
        ids.append(next_id)
    return tokenizer.decode(ids)

print(generate("h"))

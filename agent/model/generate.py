import torch
import torch.nn.functional as F
from model import TinyGPT
from tokenizer import CharTokenizer

block_size = 64

text = open("data/train.txt").read()
tokenizer = CharTokenizer(text)

model = TinyGPT(tokenizer.vocab_size, block_size=block_size)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def sample(logits, temperature=1.0, top_k=10):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, top_k)
    probs = F.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1)
    return indices[idx]

def generate(start, length=100, temperature=0.8, top_k=5):
    ids = tokenizer.encode(start)

    for _ in range(length):
        x = torch.tensor([ids[-block_size:]])
        logits = model(x)
        next_id = sample(logits[0, -1], temperature, top_k)
        ids.append(next_id.item())

    return tokenizer.decode(ids)

prompt = """### Instruction:
사과 바나나 포도를 순서대로 처리해라

### Response:
"""

print(generate(prompt, temperature=0.7, top_k=5))

import torch
import math
from model import TinyGPT
from tokenizer import CharTokenizer

block_size = 64
batch_size = 16
steps = 10000
lr = 3e-4

text = open("data/train.txt").read()
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

model = TinyGPT(tokenizer.vocab_size, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()

for step in range(steps):
    idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])

    logits = model(x)
    loss = loss_fn(logits.view(-1, tokenizer.vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, "loss:", loss.item(), "ppl:", math.exp(loss.item()))

torch.save(model.state_dict(), "model.pt")

print("model saved")

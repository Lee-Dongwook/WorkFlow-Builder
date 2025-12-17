import torch
from model import TinyGPT
from tokenizer import CharTokenizer

block_size = 32

text = open("data/train.txt").read()
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

model = TinyGPT(tokenizer.vocab_size, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(5000):
    idx = torch.randint(0, len(data) - block_size - 1, (1,))
    x = data[idx:idx + block_size].unsqueeze(0)
    y = data[idx + 1:idx + block_size + 1].unsqueeze(0)

    logits = model(x)
    loss = loss_fn(logits.view(-1, tokenizer.vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())

torch.save(model.state_dict(), "model.pt")

print("Training complete!")

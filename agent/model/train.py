import torch
import math
from model import TinyGPT
from tokenizer import CharTokenizer

block_size = 64
batch_size = 16
steps = 10000
lr = 3e-4

train_text = open("data/train.txt").read()
eval_text = open("data/eval.txt").read()
tokenizer = CharTokenizer(train_text + eval_text)
train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
eval_data = torch.tensor(tokenizer.encode(eval_text), dtype=torch.long)

model = TinyGPT(tokenizer.vocab_size, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    
    for _ in range(50):
        idx = torch.randint(0, len(eval_data) - block_size - 1, (batch_size,))
        x = torch.stack([eval_data[i:i+block_size] for i in idx])
        y = torch.stack([eval_data[i+1:i+block_size+1] for i in idx])

        logits = model(x)
        loss = loss_fn(
            logits.view(-1, tokenizer.vocab_size),
            y.view(-1)
        )
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


for step in range(steps):
    idx = torch.randint(0, len(train_data) - block_size - 1, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in idx])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in idx])

    logits = model(x)
    loss = loss_fn(logits.view(-1, tokenizer.vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        eval_loss = evaluate()
        print(
            f"step {step} | "
            f"train ppl {math.exp(loss.item()):.2f} | "
            f"eval ppl {math.exp(eval_loss):.2f}"
        )

torch.save(model.state_dict(), "model.pt")

print("model saved")

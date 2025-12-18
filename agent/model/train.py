import torch
import math
from torch.amp import autocast, GradScaler
from model import TinyGPT
from tokenizer.bpe_class import BPETokenizer

block_size = 64
batch_size = 16
total_steps = 10000
eval_interval = 500

max_lr = 3e-4
min_lr = 3e-5
warmup_steps = 500

max_grad_norm = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_text = open("data/train.txt").read()
eval_text = open("data/eval.txt").read()
tokenizer = BPETokenizer.load("tokenizer/")

train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long, device=device)
eval_data = torch.tensor(tokenizer.encode(eval_text), dtype=torch.long, device=device)

model = TinyGPT(tokenizer.vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


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

model.train()
scaler = GradScaler()

for step in range(total_steps):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    idx = torch.randint(0, len(train_data) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([train_data[i:i+block_size] for i in idx])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in idx])

    with autocast():
        logits, _ = model(x, None)
        loss = loss_fn(logits.view(-1, tokenizer.vocab_size), y.view(-1))

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0:
        eval_loss = evaluate()
        print(
            f"step {step:5d} | "
            f"lr {lr:.2e} | "
            f"train ppl {math.exp(loss.item()):7.2f} | "
            f"eval ppl {math.exp(eval_loss):7.2f} | "
        )

torch.save(model.state_dict(), "model.pt")
print("model saved")

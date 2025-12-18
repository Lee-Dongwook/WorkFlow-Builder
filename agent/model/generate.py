import torch
import torch.nn.functional as F
from model import TinyGPT
from tokenizer.bpe_class import BPETokenizer

block_size = 64

tokenizer = BPETokenizer.load("tokenizer/")
model = TinyGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("model.pt"))
model.eval()

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_p=0.9):
    model.eval()

    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    cache = None

    logits, cache = model(idx, cache)

    for _ in range(max_new_tokens):
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(
            probs, descending=True
        )
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True) 

        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_idx = torch.gather(sorted_idx, -1, next_token)

        if next_idx.item() == tokenizer.eos_id:
            break

        idx = torch.cat([idx, next_idx], dim=1)
        logits, cache = model(next_idx, cache)

    return tokenizer.decode(idx[0].tolist(), skip_special=True)

prompt = """### Instruction:
사과 바나나 포도를 순서대로 처리해라

### Response:
"""

print(generate(model, tokenizer, "### Instruction:\n순서를 지켜서 사과 바나나 포도 실행\n\n### Response:\n", temperature=0.8, top_p=0.9))

from generate import generate
from tokenizer.bpe_class import BPETokenizer
from model import TinyGPT
import torch

tokenizer = BPETokenizer.load("tokenizer/")
model = TinyGPT(tokenizer.vocab_size)
model.load_state_dict(torch.load("model.pt"))
model.eval()

prompt = """### Instruction:
다음 형식으로 새로운 instruction을 하나 생성해라.

- 작업은 반드시 순서가 있는 작업
- 예: A → B → C
- 출력은 Response에만 작성

### Response:
"""

samples = []

for _ in range(100):
    out = generate(
        model,
        tokenizer,
        prompt,
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=80
    )

    samples.append(out)

with open("data/self_raw.txt", "w") as f:
    f.write("\n\n".join(samples))

print("self generated:", len(samples))

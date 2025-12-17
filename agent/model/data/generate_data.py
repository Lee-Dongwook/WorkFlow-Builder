import random

actions = [
    ["사과", "바나나", "포도"],
    ["로그인", "결제", "완료"],
    ["step1", "step2", "step3"],
    ["A", "B", "C"]
]

templates = [
    "다음 작업을 순서대로 실행해라: {items}",
    "{items}를 차례대로 처리해라",
    "순서를 지켜서 {items} 실행"
]

def make_example(items):
    instruction = random.choice(templates).format(
        items=" ".join(items)
    )
    response = " → ".join(items)

    return f"""### Instruction:
{instruction}

### Response:
{response}
"""

data = []

for _ in range(300):
    items = random.choice(actions)
    data.append(make_example(items))

random.shuffle(data)
split = int(len(data) * 0.9)
train_data = data[:split]
eval_data = data[split:]

with open("data/train.txt", "w") as f:
    f.write("\n".join(train_data))

with open("data/eval.txt", "w") as f:
    f.write("\n".join(eval_data))

print("train:", len(train_data), "eval:", len(eval_data))


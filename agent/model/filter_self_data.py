import re

raw = open("data/self_raw.txt").read().split("\n\n")
clean = []

pattern = re.compile(
    r"### Instruction:\n(.+?)\n\n### Response:\n(.+)",
    re.S
)

for block in raw:
    m = pattern.search(block)
    if not m:
        continue

    inst, resp = m.group(1).strip(), m.group(2).strip()

    if "→" not in resp:
        continue
    if len(inst) < 10:
        continue

    clean.append(f"""### Instruction:
{inst}

### Response:
{resp}
""")

with open("data/self_clean.txt", "w") as f:
    f.write("\n".join(clean))

print("clean:", len(clean))

import json
import random

FILES = [
    "dataset/brainrot.jsonl",
    "dataset/brainrot_generated.jsonl",
]

OUTPUT = "dataset/brainrot_combined.jsonl"

dataset = []

for filename in FILES:
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

random.shuffle(dataset)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("=" * 50)
print("COMBINED DATASET")
print("=" * 50)
print(f"Total examples: {len(dataset)}")
print(f"Saved to: {OUTPUT}")
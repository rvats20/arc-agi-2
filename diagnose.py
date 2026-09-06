import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arc_agi2 import load_all

DATA = Path(os.environ.get("ARC_DATA_DIR", HERE / "data"))
tasks = load_all(DATA, split="evaluation")

ident = 0
n_train1 = 0
for t in tasks.values():
    if len(t.train) >= 1:
        if len(t.train) == 1:
            n_train1 += 1
        for p in t.train:
            if p["input"] == p["output"]:
                ident += 1

nt = sum(len(t.test) for t in tasks.values())
sizes = set()
for t in tasks.values():
    for p in t.train + t.test:
        g = p["input"]
        sizes.add((len(g), len(g[0]) if g else 0))

print("eval tasks:", len(tasks), "| tasks-with-1-train:", n_train1)
print("train pairs where input==output (identity would solve):", ident)
print("total test inputs:", nt)
print("distinct grid dims (first 8):", sorted(sizes)[:8], "| total distinct:", len(sizes))

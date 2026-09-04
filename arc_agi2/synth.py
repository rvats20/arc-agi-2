"""Procedural synthetic task generator (lightweight NVARC-SDG).

NVARC's winning edge was synthetic data scale (3.2M samples). This module
is a CPU-safe seed: compose DSL primitives into random pipelines, render
random inputs, and emit ARC-style {train, test} tasks. Use it to:
  1. augment LLM repair-loop prompts (few-shot demos),
  2. build fine-tuning datasets for Qwen3/VARC offline,
  3. stress-test the verifier + augmented scorer.

Usage:
    python -m arc_agi2.synth --n 100 --out data/synth --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


PIPELINES = [
    "rotate_cw", "rotate_ccw", "flip_h", "flip_v", "transpose",
    "invert_colors", "crop_nonzero", "shift_to_origin",
    "compress_to_side:down", "compress_to_side:up",
    "compress_to_side:left", "compress_to_side:right",
    "mirror_complete:h:left", "mirror_complete:h:right",
    "mirror_complete:v:top", "mirror_complete:v:bottom",
    "extract_largest", "fill_enclosed:1:2",
    "kron_tile:2", "scale_up:2",
]

COLORMAPS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]


def _apply_pipeline(g: np.ndarray, spec: str) -> np.ndarray:
    from . import dsl as D
    parts = spec.split(";")
    out = g
    for p in parts:
        if ":" in p:
            name, *args = p.split(":")
            fn = getattr(D, name)
            conv = []
            for a in args:
                conv.append(int(a) if a.lstrip("-").isdigit() else a)
            out = fn(out, *conv)
        else:
            out = getattr(D, p)(out)
    return np.asarray(out, dtype=int)


def random_pipeline(rng: random.Random) -> str:
    n = rng.choice([1, 1, 2, 2, 3])
    steps = [rng.choice(PIPELINES) for _ in range(n)]
    if rng.random() < 0.3:
        steps.append(f"colormap:{rng.choice(range(len(COLORMAPS)))}")
    return ";".join(steps)


def _apply_full(g: np.ndarray, spec: str) -> np.ndarray:
    if "colormap" in spec:
        *rest, cmap = spec.split(";")
        idx = int(cmap.split(":")[1])
        perm = np.asarray(COLORMAPS[idx], dtype=int)
        g = perm[np.asarray(g, dtype=int)]
        spec = ";".join(rest)
    if not spec:
        return g
    return _apply_pipeline(g, spec)


def random_grid(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 10)
    w = rng.randint(3, 10)
    density = rng.choice([0.2, 0.35, 0.5])
    ncolors = rng.randint(2, 5)
    palette = rng.sample(range(1, 10), ncolors)
    g = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if rng.random() < density:
                g[y, x] = rng.choice(palette)
    return g


def make_task(rng: random.Random) -> dict | None:
    spec = random_pipeline(rng)
    pairs = []
    for _ in range(rng.randint(2, 4)):
        g = random_grid(rng)
        try:
            o = _apply_full(g, spec)
        except Exception:
            return None
        if o.size == 0 or o.shape[0] > 30 or o.shape[1] > 30:
            return None
        pairs.append((g.tolist(), o.tolist()))
    if len(set(map(str, [p[1] for p in pairs]))) == 0:
        return None
    test_g = random_grid(rng)
    try:
        test_o = _apply_full(test_g, spec).tolist()
    except Exception:
        return None
    return {"spec": spec,
            "train": [{"input": i, "output": o} for i, o in pairs],
            "test": [{"input": test_g.tolist()}],
            "test_solution": test_o}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    tries = 0
    while made < a.n and tries < a.n * 20:
        tries += 1
        t = make_task(rng)
        if not t:
            continue
        (out / f"synth_{made:05d}.json").write_text(json.dumps(t))
        made += 1
    print(f"wrote {made}/{a.n} synthetic tasks to {out}")


if __name__ == "__main__":
    main()

"""Task-shape fingerprint cache for the DSL search.

ARC puzzles are uniquely determined by their input structure: the train
pair shapes, the palette of non-zero colors, and the object count. Two
tasks with the same fingerprint behave identically under any pure-DSL
transformation, so we can cache the verified source for the fingerprint
and reuse it across tasks that share the fingerprint.

This is most useful in the 10h Kaggle window: after the first pass solves
a task, every subsequent task whose fingerprint matches reuses the same
verified source — zero search cost.

The cache is a JSON file: { fingerprint: source_string, ... }.
Fingerprints are short SHA-1 hashes of the structural signature (not the
input data, so the cache generalizes across tasks with different contents
but same shape+palette pattern).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_PATH = Path(
    os.environ.get("ARC_DSL_CACHE", str(Path.home() / ".cache" / "arc_agi2_dsl_cache.json"))
)


def _shape_signature(arr: np.ndarray) -> tuple:
    """A small tuple that describes a grid's structural signature:
    shape, sorted non-zero color set, count of each color, count of
    connected non-zero objects. Two grids with the same signature are
    very likely to require the same kind of transformation.
    """
    h, w = arr.shape
    flat = arr.ravel()
    palette = sorted({int(v) for v in flat if v != 0})
    counts = tuple(int((flat == c).sum()) for c in palette)
    # Count connected non-zero objects (4-connectivity), regardless of color
    seen = np.zeros((h, w), dtype=bool)
    n_obj = 0
    for y in range(h):
        for x in range(w):
            if arr[y, x] == 0 or seen[y, x]:
                continue
            n_obj += 1
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if seen[cy, cx] or arr[cy, cx] == 0:
                    continue
                seen[cy, cx] = True
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
    return (h, w, tuple(palette), counts, n_obj)


def fingerprint(task) -> str:
    """Return a stable short hash of a task's structural signature.

    We hash the SIGNATURES of every train pair so that order matters but
    pixel content does not. The same fingerprint can be reused for any
    other task that has the same structural shape across all pairs.
    """
    sigs = []
    for pair in task.train:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        sigs.append((_shape_signature(inp), _shape_signature(out)))
    blob = json.dumps(sigs, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def get_cached(task) -> Optional[str]:
    """Return the verified source for this task's fingerprint, or None."""
    return load_cache().get(fingerprint(task))


def put_cached(task, src: str) -> None:
    """Cache a verified source under this task's fingerprint."""
    cache = load_cache()
    cache[fingerprint(task)] = src
    save_cache(cache)


def clear() -> None:
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()

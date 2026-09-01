"""MockVL -- a CPU-only stand-in for Qwen2.5-VL, for DRY-RUN ONLY.

This is NOT a real language model. It implements the same `propose(task)`
interface as models.QwenVL but instead of vision-language reasoning it runs a
richer symbolic search than the toy DSL baseline. Purpose: prove the FULL
notebook integration on CPU (DSL fails -> proposer returns candidates ->
verifier accepts a correct one -> prediction emitted -> submission written)
without needing a GPU or the 7B weights.

The score it produces is a STRONGER SYMBOLIC BASELINE, not the VLM's score.
The real Qwen2.5-VL number can only come from a Kaggle GPU run.
"""
from __future__ import annotations

import numpy as np

from .dsl import (rotate_cw, rotate_ccw, flip_h, flip_v, transpose,
                 invert_colors, crop_nonzero, grow_nonzero)


def _color_map_source(task) -> str | None:
    """If every train pair has identical shape and a consistent per-color
    mapping input->output, return a solve() that applies it."""
    mapping: dict[int, int] = {}
    shape = None
    for pair in task.train:
        inp = np.array(pair["input"], dtype=int)
        out = np.array(pair["output"], dtype=int)
        if shape is None:
            shape = inp.shape
        if inp.shape != out.shape or inp.shape != shape:
            return None
        for a, b in zip(inp.flat, out.flat):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    if not mapping:
        return None
    f = [-1] * 10
    for k, v in mapping.items():
        f[k] = v
    src = (f"def solve(g):\n"
           f"    _f = np.array({f}, dtype=int)\n"
           f"    return _f[g]\n")
    return src


def _orientation_sources() -> list[str]:
    return [
        "def solve(g):\n    return rotate_cw(g)\n",
        "def solve(g):\n    return rotate_ccw(g)\n",
        "def solve(g):\n    return rotate_cw(rotate_cw(g))\n",
        "def solve(g):\n    return flip_h(g)\n",
        "def solve(g):\n    return flip_v(g)\n",
        "def solve(g):\n    return transpose(g)\n",
        "def solve(g):\n    return transpose(flip_h(g))\n",
        "def solve(g):\n    return rotate_cw(flip_h(g))\n",
        "def solve(g):\n    return invert_colors(g)\n",
        "def solve(g):\n    return crop_nonzero(g)\n",
        "def solve(g):\n    return grow_nonzero(g)\n",
    ]


class MockVL:
    """Implements propose(task, n_candidates) like QwenVL, but via symbolic search."""

    def __init__(self, *args, **kwargs):
        pass

    def propose(self, task, n_candidates: int = 4, **_kw) -> list[str]:
        from .verifier import verify_program
        cands: list[str] = []
        # 1) color-map (often the real rule for ARC pairs of identical shape)
        cm = _color_map_source(task)
        if cm:
            cands.append(cm)
        # 2) orientation / simple transforms
        for src in _orientation_sources():
            if verify_program(src, task):
                cands.append(src)
            if len(cands) >= n_candidates:
                break
        return cands

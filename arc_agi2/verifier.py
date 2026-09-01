"""Verifier: run a candidate program against train pairs, check exact match.

A "program" is a Python callable (grid -> grid) SOURCED from a restricted
namespace. This module provides:
  - run_program(src, grid): eval a DSL string safely and apply it to a grid
  - verify_program(src, task): apply to all train inputs; True iff every
                                output matches the train output exactly.

The DSL string is what the LLM/VLM proposes; the verifier is the symbolic
gate that must pass before we trust the program on the test input.
"""
from __future__ import annotations

import numpy as np

from .dsl import SAFE_GLOBALS


def run_program(src: str, grid) -> list[list[int]]:
    """Apply a DSL source string to a single grid. Returns list[list[int]]."""
    g = np.array(grid, dtype=int)
    namespace: dict = {}
    exec(compile(src, "<program>", "exec"), dict(SAFE_GLOBALS), namespace)
    if "solve" not in namespace:
        raise ValueError("program must define solve(g: ndarray) -> ndarray")
    out = namespace["solve"](g)
    out = np.array(out, dtype=int)
    return out.tolist()


def verify_program(src: str, task, verbose: bool = False) -> bool:
    """Return True iff the program reproduces every train output exactly."""
    try:
        for i, pair in enumerate(task.train):
            pred = run_program(src, pair["input"])
            gold = np.array(pair["output"], dtype=int)
            if not np.array_equal(np.array(pred), gold):
                if verbose:
                    print(f"  train pair {i}: MISMATCH")
                return False
        return True
    except Exception as e:  # malformed program -> fails verification
        if verbose:
            print(f"  program error: {type(e).__name__}: {e}")
        return False

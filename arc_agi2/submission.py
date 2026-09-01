"""Submission writing for ARC-AGI-2.

Format (official spec):
  submission.json = {
     "<task_id>": [ {"attempt_1": grid, "attempt_2": grid}, ... ],  # one per test input
  }
  Each grid is list[list[int]] with ints 0-9.
  BOTH attempt_1 and attempt_2 required for every test input, even if dummy.
"""
from __future__ import annotations

import json
from pathlib import Path


def empty_submission(task_ids: list[str], n_test_per_id: dict[str, int]) -> dict:
    """Build a submission skeleton filled with empty 1x1 grids (valid dummy)."""
    sub = {}
    for tid in task_ids:
        sub[tid] = [
            {"attempt_1": [[0]], "attempt_2": [[0]]}
            for _ in range(n_test_per_id.get(tid, 1))
        ]
    return sub


def write_submission(path, task_predictions: dict[str, list[list[list[int]]]]):
    """task_predictions: tid -> list of predictions, one per test input.
    Each prediction is a 2-tuple/list [attempt_1_grid, attempt_2_grid].
    Writes submission.json in the official format. Returns the dict."""
    sub = {}
    for tid, preds in task_predictions.items():
        sub[tid] = []
        for p in preds:
            a1, a2 = p[0], p[1]
            sub[tid].append({"attempt_1": a1, "attempt_2": a2})
    Path(path).write_text(json.dumps(sub))
    return sub


def validate_submission(sub: dict, task_ids: list[str], n_test_per_id: dict[str, int]) -> list[str]:
    """Return a list of problems (empty if valid)."""
    errs = []
    for tid in task_ids:
        if tid not in sub:
            errs.append(f"missing task {tid}")
            continue
        preds = sub[tid]
        if len(preds) != n_test_per_id.get(tid, 1):
            errs.append(f"{tid}: expected {n_test_per_id.get(tid,1)} predictions, got {len(preds)}")
        for i, p in enumerate(preds):
            if "attempt_1" not in p or "attempt_2" not in p:
                errs.append(f"{tid} pred {i}: missing attempt keys")
                continue
            for k in ("attempt_1", "attempt_2"):
                g = p[k]
                if not (isinstance(g, list) and g and isinstance(g[0], list)):
                    errs.append(f"{tid} pred {i} {k}: not a 2D grid")
    return errs

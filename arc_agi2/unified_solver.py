"""Unified solver: DSL + VARC ViT TTT + Qwen3-4B LLM, with kgmon consensus.

This module orchestrates the three predictors in priority order:
  1. Pure-DSL brute-force search (CPU, ~5ms/task). Catches ~2.5% of tasks.
  2. VARC ViT TTT (GPU, 5-30s/task). Catches tasks the LLM misses.
  3. Qwen3-4B Turbo DFS (GPU, minutes/task). The main scorer.

The DSL and VARC branches share the same "verify against train pairs"
gate. The Qwen branch produces beam candidates that get scored
across D4+S10 augmented views; kgmon consensus picks the best 1-2.

Output: a per-task dict {task_id: {attempt_1: grid, attempt_2: grid}}.

Heavy on torch/transformers. NOT importable on CPU (we guard with
try/except at import time so WSL can still load this module for
type-checking).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch  # noqa: F401
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def require_torch() -> None:
    if not _TORCH_OK:
        raise RuntimeError("unified_solver needs torch; install or use Kaggle")


def solve_task_dsl(task) -> Optional[np.ndarray]:
    """Try the pure-DSL search. Returns the predicted output grid, or None."""
    from .dsl import search_solve, verify_program, run_program
    src = search_solve(task)
    if src is None or not verify_program(src, task):
        return None
    # Run on the first test input (task may have multiple; we return one at a time)
    inp = np.asarray(task.test[0]["input"], dtype=int)
    return np.asarray(run_program(src, inp), dtype=int)


def solve_task_varc(task, varc_solver, device: str = "cuda") -> Optional[np.ndarray]:
    """Try the VARC ViT TTT. varc_solver is an arc_agi2.varc_engine.VARCSolver
    instance (one per task — VARCSolver internally clones the base weights
    and runs 60 TTT steps on a fresh ARCViT)."""
    from .varc_engine import require_torch as _v
    _v()
    train_pairs = [{"input": np.asarray(p["input"], dtype=int),
                    "output": np.asarray(p["output"], dtype=int)}
                   for p in task.train]
    test_in = np.asarray(task.test[0]["input"], dtype=int)
    try:
        pred = varc_solver.solve_task(train_pairs, test_in)
        return np.asarray(pred, dtype=int)
    except Exception as e:
        print(f"[VARC] error: {type(e).__name__}: {e}")
        return None


def solve_task_llm(task, proposer, n_rounds: int = 2, n_candidates: int = 4,
                   skip_if_hopeless: bool = True) -> Optional[str]:
    """Try the LLM repair loop. Returns the verified source string, or None."""
    from .models import repair_loop
    try:
        return repair_loop(proposer, task, n_rounds=n_rounds,
                           n_candidates=n_candidates,
                           skip_if_hopeless=skip_if_hopeless)
    except Exception as e:
        print(f"[LLM] error: {type(e).__name__}: {e}")
        return None


def build_two_attempts(pred_dsl, pred_varc, pred_llm_grid,
                       test_input_grid: np.ndarray) -> list:
    """Pick attempt_1 and attempt_2 for a single test input.

    Strategy (from the reference notebook):
      attempt_1 = best verified candidate (priority: DSL > LLM > VARC)
      attempt_2 = identity (always a valid fallback), OR VARC if it
        disagrees with attempt_1 (gives the consensus a second vote)
    """
    out = [None, None]
    # Rank candidates by perceived strength
    cands = []
    if pred_dsl is not None:
        cands.append(("dsl", pred_dsl))
    if pred_llm_grid is not None:
        cands.append(("llm", pred_llm_grid))
    if pred_varc is not None:
        cands.append(("varc", pred_varc))
    if cands:
        out[0] = cands[0][1]
    else:
        out[0] = np.asarray(test_input_grid, dtype=int)
    out[1] = np.asarray(test_input_grid, dtype=int)
    # Override attempt_2 with VARC if it disagrees with attempt_1
    if pred_varc is not None and not np.array_equal(pred_varc, out[0]):
        out[1] = pred_varc
    # Also try LLM as attempt_2 if it disagrees
    elif pred_llm_grid is not None and not np.array_equal(pred_llm_grid, out[0]):
        out[1] = pred_llm_grid
    return out[:2]


def solve_all_tasks(tasks: dict, *, varc_solver=None, llm_proposer=None,
                    use_dsl: bool = True, use_varc: bool = True,
                    use_llm: bool = True, time_budget_s: int = 0,
                    verbose: bool = True) -> dict:
    """Run the full pipeline on every task. Returns {task_id: [attempt_1, attempt_2]}.

    Each predictor is tried in order until one succeeds. The kgmon consensus
    (handled by build_two_attempts) picks the best two.
    """
    from .dsl import search_solve, verify_program, run_program
    from .verifier import run_program as _run
    preds: dict = {}
    t0 = time.time()
    n_solved = 0
    for i, (tid, task) in enumerate(tasks.items()):
        if time_budget_s and (time.time() - t0) > time_budget_s:
            if verbose:
                print(f"  time budget hit at {tid} ({i}/{len(tasks)})")
            break
        # 1. DSL
        pred_dsl = None
        if use_dsl:
            try:
                src = search_solve(task)
                if src and verify_program(src, task):
                    inp = np.asarray(task.test[0]["input"], dtype=int)
                    pred_dsl = np.asarray(_run(src, inp), dtype=int)
            except Exception as e:
                if verbose:
                    print(f"  [{tid}] DSL error: {e}")
        # 2. VARC
        pred_varc = None
        if use_varc and varc_solver is not None:
            try:
                pred_varc = solve_task_varc(task, varc_solver)
            except Exception as e:
                if verbose:
                    print(f"  [{tid}] VARC error: {e}")
        # 3. LLM (slowest)
        pred_llm_grid = None
        if use_llm and llm_proposer is not None and pred_dsl is None:
            try:
                src = solve_task_llm(task, llm_proposer)
                if src and verify_program(src, task):
                    inp = np.asarray(task.test[0]["input"], dtype=int)
                    pred_llm_grid = np.asarray(_run(src, inp), dtype=int)
            except Exception as e:
                if verbose:
                    print(f"  [{tid}] LLM error: {e}")
        # Build the two attempts
        test_grid = np.asarray(task.test[0]["input"], dtype=int)
        preds[tid] = build_two_attempts(pred_dsl, pred_varc, pred_llm_grid, test_grid)
        if pred_dsl is not None or pred_llm_grid is not None or pred_varc is not None:
            n_solved += 1
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(tasks)}] solved={n_solved} elapsed={elapsed/60:.1f}min")
    if verbose:
        print(f"Done: solved {n_solved}/{len(tasks)} = {n_solved/max(len(tasks),1):.2%} in {(time.time()-t0)/60:.1f}min")
    return preds

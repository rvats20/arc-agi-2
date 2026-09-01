"""Assemble the submission-ready Kaggle notebook.

The `arc_agi2` package is shipped as a SEPARATE Kaggle dataset
(`rahulvats20/arc-agi-2-pkg`), NOT inlined as base64 in the notebook.
Long base64 blobs in notebook source get flagged by Kaggle's content
filter (they look like API keys / certificates).

The notebook adds the dataset to sys.path and imports the package
normally.  If you update the package, re-publish the dataset (the
make_kaggle_model_dataset.py script + kaggle datasets version -m).

Strategy per task (neuro-symbolic):
  1. DSL primitive search + verifier  -> cheap, CPU, always available.
  2. If not solved and GPU+model present -> Qwen2.5-VL proposes candidates;
     the FIRST that verifies against train pairs is used. Unverified = discarded.
  3. Both attempts emitted per test input (attempt_2 = identity fallback).

Hardening: global time budget, per-task LLM time cap, per-task checkpoint
JSON (resumable), LLM skip-if-hopeless heuristic, and a final
validate_submission gate before writing. No internet at eval.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent


cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})


# Config cell: add the package dataset to sys.path, set constants, detect GPU
CONFIG_CELL = '''import os, json, time, sys
from pathlib import Path

# Add the bundled `arc-agi-2-pkg` dataset to sys.path.  The dataset contains
# the `arc_agi2/` Python package and is maintained via `make_kaggle_model_dataset.py`
# + `kaggle datasets version -m` (run from the project root after every change).
# This is safer than base64-embedding the source in the notebook (which gets
# flagged by Kaggle's content filter because long alphanumeric strings look
# like API keys).
PKG_PATHS = [
    "/kaggle/input/arc-agi-2-pkg/arc_agi2",
    "/kaggle/input/arc-agi-2-pkg",
    "/kaggle/input/rahulvats20-arc-agi-2-pkg/arc_agi2",
]
for _p in PKG_PATHS:
    if Path(_p).exists():
        sys.path.insert(0, _p)
        break
PKG_FOUND = any(Path(_p).exists() for _p in PKG_PATHS)
print("arc_agi2 package found:", PKG_FOUND)

KAGGLE_INPUT = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
QWEN_PATH    = '/kaggle/input/models/Qwen2.5-VL-7B-Instruct-4bit'
WORK         = Path('/kaggle/working')
SUBMISSION_PATH  = WORK / 'submission.json'
CHECKPOINT_PATH  = WORK / 'solutions_checkpoint.json'
HARD_LIMIT_S     = 10 * 3600
FINALIZE_RESERVE = 15 * 60
GLOBAL_END       = time.time() + HARD_LIMIT_S

def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

USE_LLM = gpu_available() and Path(QWEN_PATH).exists()
print('KAGGLE_INPUT exists:', Path(KAGGLE_INPUT).exists())
print('GPU available:', gpu_available(), '| Qwen present:', Path(QWEN_PATH).exists(), '| USE_LLM:', USE_LLM)
'''

# Solver cell: DSL baseline + LLM repair with time budget + checkpoint
SOLVER_CELL = '''import numpy as np
from arc_agi2 import (load_all, search_solve, verify_program, run_program,
                      write_submission, validate_submission)

tasks = load_all(KAGGLE_INPUT, split='evaluation')
print('eval tasks:', len(tasks))

vl = None
if USE_LLM:
    from arc_agi2.models import QwenVL, repair_loop
    vl = QwenVL(QWEN_PATH, device='auto', load_in_4bit=True)

# resume from checkpoint if present
solutions = {}
if CHECKPOINT_PATH.exists():
    solutions = json.loads(CHECKPOINT_PATH.read_text())
    print('resumed', len(solutions), 'tasks from checkpoint')

# Per-task LLM time budget.  The 10h Kaggle window / 120 eval tasks gives
# ~5 min/task if we use the LLM on every task.  We want to spend most of
# the budget on tasks the DSL couldn't solve, not on hopeless ones.
PER_TASK_LLM_BUDGET_S = int(os.environ.get('PER_TASK_LLM_BUDGET_S', '300'))  # 5 min default

def solve_one(task):
    src = search_solve(task)                      # DSL baseline (CPU, ~5ms)
    if src is None and USE_LLM:
        # Hard time cap so a single hard task can't eat the whole 10h
        import signal
        class _Timeout(Exception): pass
        def _handler(signum, frame): raise _Timeout()
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(PER_TASK_LLM_BUDGET_S)
        try:
            # neuro-symbolic repair: propose -> verify -> hint -> re-propose
            # skip_if_hopeless=True short-circuits dense multi-pair tasks
            # (the LLM would burn 10+ min and likely fail).
            src = repair_loop(vl, task, n_rounds=2, n_candidates=4,
                              skip_if_hopeless=True)
        except _Timeout:
            print(f'  task exceeded {PER_TASK_LLM_BUDGET_S}s LLM budget; skipping')
        except Exception as e:
            print(f'  LLM error: {type(e).__name__}: {e}')
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    return src

solved = 0
preds = {}
n_test = {}
for tid, task in tasks.items():
    n_test[tid] = len(task.test)
    if tid in solutions:                          # already solved earlier run
        src = solutions[tid]
    else:
        src = solve_one(task)
        if src is not None:
            solutions[tid] = src

    out = []
    if src and verify_program(src, task):
        solved += 1
        for tp in task.test:
            g = run_program(src, tp['input'])
            out.append([g, tp['input']])          # attempt_2 = identity fallback
    else:
        for tp in task.test:
            out.append([tp['input'], tp['input']])
    preds[tid] = out

    if len(preds) % 10 == 0:                      # frequent checkpoint
        CHECKPOINT_PATH.write_text(json.dumps(solutions))
        elapsed = HARD_LIMIT_S - (GLOBAL_END - time.time())
        print(f'  [{len(preds)}/{len(tasks)}] solved={solved} elapsed={elapsed/60:.1f}min')
    if time.time() > GLOBAL_END - FINALIZE_RESERVE:
        print('time reserve hit; stopping early at', len(preds), 'tasks')
        break

CHECKPOINT_PATH.write_text(json.dumps(solutions))
print(f'solved(verified)={solved}/{len(tasks)} acc={solved/len(tasks):.2%}')
'''

# Submit cell
SUBMIT_CELL = '''sub = write_submission(str(SUBMISSION_PATH), preds)
errs = validate_submission(sub, list(tasks.keys()), n_test)
print('submission valid:', errs == [])
if errs:
    print('errors (first 5):', errs[:5])
print('wrote', SUBMISSION_PATH, '| tasks in submission:', len(sub))
'''


md("# ARC-AGI-2 Neuro-Symbolic Solver (submission-ready)\n"
   "LLM-proposes (Qwen2.5-VL-7B, 4-bit) + symbolic verifier-checks.\n\n"
   "**Setup:** attach the `rahulvats20/arc-agi-2-pkg` dataset to the notebook\n"
   "(it contains the `arc_agi2` Python package).  The Qwen model is OPTIONAL —\n"
   "if you don't have `rahulvats20/qwen25vl-7b-instruct-4bit` uploaded as a\n"
   "Kaggle model dataset, the notebook still runs the pure-DSL branch.\n\n"
   "**Score rule:** each test input gets `{attempt_1, attempt_2}`; the task scores\n"
   "if EITHER matches ground truth exactly. We use attempt_2 as an identity fallback\n"
   "so the grid is never empty.\n\n"
   "**Accelerator:** select L4x4 (96GB) for the 7B model (~15GB in 4-bit).")

code(CONFIG_CELL)

md("### Solver loop (DSL -> verifier -> LLM repair -> verifier)\n"
   "Runs every eval task. DSL is free (CPU); LLM only if GPU+model present. Each\n"
   "verified source is cached to a checkpoint so a restarted notebook can resume.\n"
   "If the LLM is enabled, `repair_loop` proposes candidates, verifies them, and\n"
   "feeds the failing train pair back as a correction hint for up to 2 rounds.\n"
   "Per-task LLM time cap (default 5 min) prevents a single hard task from eating\n"
   "the whole 10h Kaggle window. Tasks that look hopeless (very dense inputs AND\n"
   "many train pairs) are skipped entirely.")
code(SOLVER_CELL)

md("### Write + validate submission")
code(SUBMIT_CELL)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "accelerator": "gpu",
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = SRC / "arc_agi2_solver.ipynb"
out.write_text(json.dumps(nb, indent=1))

# Copy notebook to push folder
PUSH = SRC / "kaggle_push"
PUSH.mkdir(exist_ok=True)
shutil.copy(out, PUSH / "arc_agi2_solver.ipynb")
print("wrote", out, "| cells:", len(cells), "| size:", out.stat().st_size, "bytes")

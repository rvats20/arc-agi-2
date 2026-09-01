"""Assemble the submission-ready Kaggle notebook (self-contained, GPU-guarded).

Strategy per task (neuro-symbolic):
  1. DSL primitive search + verifier  -> cheap, CPU, always available.
  2. If not solved and GPU+model present -> Qwen2.5-VL proposes candidates;
     the FIRST that verifies against train pairs is used. Unverified = discarded.
  3. Both attempts emitted per test input (attempt_2 = identity fallback).

Hardening: global time budget, per-task checkpoint JSON (resumable), and a
final validate_submission gate before writing. No internet at eval.
"""
from __future__ import annotations
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
LIB = SRC / "arc_agi2"
MODULE_ORDER = ["loader", "grid_utils", "dsl", "verifier", "submission", "models"]


def read_module(name: str) -> str:
    return (LIB / f"{name}.py").read_text()


cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})


CONFIG_CELL = '''import os, json, time, sys, base64
from pathlib import Path

# Self-contained: (re)create the arc_agi2 package under /kaggle/working from the
# embedded sources below, so the notebook needs NO external dataset mount. We
# ALWAYS overwrite (a stale partial package from a prior run would otherwise be
# imported as a broken namespace package).
_PKG_DIR = Path("/kaggle/working/arc_agi2")
import shutil as _shutil
if _PKG_DIR.exists():
    _shutil.rmtree(_PKG_DIR)
_PKG_DIR.mkdir(parents=True, exist_ok=True)
_MODULES = {__ARC_MODULES__}
for _name, _b64 in _MODULES.items():
    (_PKG_DIR / _name).write_text(base64.b64decode(_b64).decode("utf-8"))
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))
import importlib as _il
_sys = __import__("sys")
if "arc_agi2" in _sys.modules:
    del _sys.modules["arc_agi2"]
PKG_FOUND = (_PKG_DIR / "__init__.py").exists()

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
print('PKG found:', PKG_FOUND, '| KAGGLE_INPUT exists:', Path(KAGGLE_INPUT).exists())
print('GPU available:', gpu_available(), '| Qwen present:', Path(QWEN_PATH).exists(), '| USE_LLM:', USE_LLM)
'''

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

def solve_one(task):
    src = search_solve(task)                      # DSL baseline (CPU)
    if src is None and USE_LLM:
        try:
            # neuro-symbolic repair: propose -> verify -> hint -> re-propose
            src = repair_loop(vl, task, n_rounds=3, n_candidates=4)
        except Exception as e:
            print(f'  LLM error: {type(e).__name__}: {e}')
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

    if len(preds) % 20 == 0:                      # periodic checkpoint
        CHECKPOINT_PATH.write_text(json.dumps(solutions))
    if time.time() > GLOBAL_END - FINALIZE_RESERVE:
        print('time reserve hit; stopping early at', len(preds), 'tasks')
        break

CHECKPOINT_PATH.write_text(json.dumps(solutions))
print(f'solved(verified)={solved}/{len(tasks)} acc={solved/len(tasks):.2%}')
'''

SUBMIT_CELL = '''sub = write_submission(str(SUBMISSION_PATH), preds)
errs = validate_submission(sub, list(tasks.keys()), n_test)
print('submission valid:', errs == [])
if errs:
    print('errors (first 5):', errs[:5])
print('wrote', SUBMISSION_PATH, '| tasks in submission:', len(sub))
'''


md("# ARC-AGI-2 Neuro-Symbolic Solver (submission-ready)\n"
   "LLM-proposes (Qwen2.5-VL-7B, 4-bit) + symbolic verifier-checks. The `arc_agi2`\n"
   "package is shipped as files (a dataset at /kaggle/input/arc-agi-2-pkg) and\n"
   "imported normally, so the notebook runs fully offline at eval time.\n\n"
   "**Score rule:** each test input gets `{attempt_1, attempt_2}`; the task scores\n"
   "if EITHER matches ground truth exactly. We use attempt_2 as an identity fallback\n"
   "so the grid is never empty.\n\n"
   "**Accelerator:** select L4x4 (96GB) for the 7B model (~15GB in 4-bit).")

# --- embed the arc_agi2 package as base64 so the notebook is fully self-contained
import base64 as _b64
_modules_b64 = {}
for _m in MODULE_ORDER + ["__init__"]:
    _fname = _m + ".py"
    _modules_b64[_fname] = _b64.b64encode((LIB / _fname).read_bytes()).decode("ascii")
_modules_literal = "{\n" + ",\n".join(
    f"    {_n!r}: {_v!r}" for _n, _v in _modules_b64.items()
) + "\n}"
CONFIG_CELL = CONFIG_CELL.replace("{__ARC_MODULES__}", _modules_literal)

code(CONFIG_CELL)

md("### Solver loop (DSL -> verifier -> LLM repair -> verifier)\n"
   "Runs every eval task. DSL is free (CPU); LLM only if GPU+model present. Each\n"
   "verified source is cached to a checkpoint so a restarted notebook can resume.\n"
   "If the LLM is enabled, `repair_loop` proposes candidates, verifies them, and\n"
   "feeds the failing train pair back as a correction hint for up to 3 rounds.")
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

# Copy notebook to push folder (no extra package files needed now).
import shutil
PUSH = SRC / "kaggle_push"
PUSH.mkdir(exist_ok=True)
shutil.copy(out, PUSH / "arc_agi2_solver.ipynb")
print("wrote", out, "| cells:", len(cells), "| embedded modules:", len(_modules_b64))

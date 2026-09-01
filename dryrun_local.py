"""Dry-run the FULL notebook integration on CPU using MockVL (no GPU, no weights).

Mirrors arc_agi2_solver.ipynb's solver loop exactly, but uses MockVL as the
'proposer' so we exercise: DSL -> MockVL.propose -> verify_program -> run_program
-> attempt_1/attempt_2 -> write_submission -> validate_submission.

This is a DRY-RUN: MockVL is symbolic search, NOT Qwen2.5-VL. The number it
reports is a stronger symbolic baseline than the toy DSL, not the VLM score.
The VLM number only exists after a real Kaggle GPU run.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arc_agi2 import (load_all, search_solve, verify_program, run_program,
                      write_submission, validate_submission)
from arc_agi2.mock_vl import MockVL

DATA = os.environ.get("ARC_DATA_DIR", "/mnt/c/Users/Rahul/arc-agi-2/data")
vl = MockVL()

tasks = load_all(DATA, split="evaluation")
print(f"eval tasks: {len(tasks)}")

def solve_one(task):
    src = search_solve(task)            # DSL baseline
    if src is None:
        for cand in vl.propose(task, n_candidates=4):   # proposer (MockVL here)
            if verify_program(cand, task):
                src = cand
                break
    return src

solved = 0
preds = {}
n_test = {}
t0 = time.time()
for tid, task in tasks.items():
    n_test[tid] = len(task.test)
    src = solve_one(task)
    out = []
    if src and verify_program(src, task):
        solved += 1
        for tp in task.test:
            g = run_program(src, tp["input"])
            out.append([g, tp["input"]])     # attempt_2 = identity fallback
    else:
        for tp in task.test:
            out.append([tp["input"], tp["input"]])
    preds[tid] = out

sub = write_submission(HERE / "submission_dryrun.json", preds)
errs = validate_submission(sub, list(tasks.keys()), n_test)
print(f"solved(verified)={solved}/{len(tasks)} acc={solved/len(tasks):.2%}  "
      f"({time.time()-t0:.1f}s)")
print("submission valid:", errs == [])
if errs:
    print("errors:", errs[:5])

# Score against known eval solutions (local, since we have them) to show the
# actual leaderboard-style number for attempt_1 OR attempt_2 matching either.
sols = load_all(DATA, split="evaluation")  # already has 'output' merged into test
correct = 0
total = 0
for tid, task in tasks.items():
    for j, tp in enumerate(task.test):
        total += 1
        gold = tp.get("output")
        if gold is None:
            continue
        a1, a2 = preds[tid][j]
        if a1 == gold or a2 == gold:
            correct += 1
print(f"FULL-SCORE vs ground truth (either attempt): {correct}/{total} "
      f"= {correct/total:.2%}")

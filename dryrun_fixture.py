"""Verify the DSL->propose->verify accept path fires using the fixture + MockVL.
Reuses the same loop as dryrun_local but builds its own fixture dir."""
import os, sys, json, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from arc_agi2 import load_all, search_solve, verify_program, run_program, write_submission, validate_submission
from arc_agi2.mock_vl import MockVL
from arc_agi2.dsl import rotate_cw, invert_colors

d = Path(tempfile.mkdtemp(prefix="arc_fix_"))
rng = np.random.default_rng(0)
inpA = rng.integers(0,5,size=(4,5)).tolist(); outA = rotate_cw(np.array(inpA)).tolist()
inpB = rng.integers(0,5,size=(3,3)).tolist(); outB = invert_colors(np.array(inpB)).tolist()
challenges = {"fx_rot":{"train":[{"input":inpA,"output":outA}],"test":[{"input":inpA}]},
              "fx_inv":{"train":[{"input":inpB,"output":outB}],"test":[{"input":inpB}]}}
solutions = {"fx_rot":[outA],"fx_inv":[outB]}
(d/"arc-agi_evaluation-challenges.json").write_text(json.dumps(challenges))
(d/"arc-agi_evaluation-solutions.json").write_text(json.dumps(solutions))

vl = MockVL()
tasks = load_all(d, split="evaluation")
solved=0; preds={}; n_test={}
for tid,task in tasks.items():
    n_test[tid]=len(task.test)
    src = search_solve(task)
    if src is None:
        for cand in vl.propose(task, n_candidates=4):
            if verify_program(cand,task):
                src=cand; break
    out=[]
    if src and verify_program(src,task):
        solved+=1
        for tp in task.test:
            g=run_program(src,tp["input"]); out.append([g,tp["input"]])
    else:
        for tp in task.test:
            out.append([tp["input"],tp["input"]])
    preds[tid]=out
print(f"FIXTURE solved(verified)={solved}/{len(tasks)}")
# score vs gold
correct=0; total=0
for tid,task in tasks.items():
    for j,tp in enumerate(task.test):
        total+=1; gold=tp.get("output")
        if gold is None: continue
        a1,a2=preds[tid][j]
        if a1==gold or a2==gold: correct+=1
print(f"FIXTURE full-score (either attempt): {correct}/{total} = {correct/total:.0%}")
print("MockVL accept-path works:", solved==2)

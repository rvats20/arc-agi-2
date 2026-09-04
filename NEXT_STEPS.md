# Session updates (2026-09-05) + next steps

## What changed this session

### 1. DSL probes (+6 solves: 25 → 31/1000 = 3.10%)
`arc_agi2/dsl.py` — 3 new primitives + 3 new probes wired into `synthesize()`:
- Primitives: `compress_to_side` (gravity), `mirror_complete`, `extract_largest`
- Probes: `_colormap_d4_source` (recolor × 8 D4 orientations),
  `_extract_object_source` (largest-object / bbox / keep-color),
  `_gravity_symmetry_source` (gravity + mirror completion)

### 2. NVARC 2025 winners research → 3 missing pieces implemented
The repo already had turbo DFS, kgmon consensus, VARC TTT, and repair loop.
Added what it lacked (NVARC 1st @ 24%, ARChitects 2nd — synthetic scale +
per-task TTT + augmented scoring is what wins; DSL alone caps ~3%):
- `arc_agi2/augment.py` (new) — D4 + color-perm views, `rank_candidates()`;
  wired into `repair_loop` so best-consensus candidate verifies first
- `arc_agi2/synth.py` (new) — procedural synthetic task generator;
  `python -m arc_agi2.synth --n 100 --out data/synth`
- `arc_agi2/ttt.py` (new) — per-task LoRA priming on own train pairs;
  GPU-only, silent no-op on CPU; called at top of `repair_loop`
- System prompts (`models.py`, `models_nvarc.py`) list all 19 primitives
- Notebook rebuilt via `build_notebook.py`

### Verified (CPU)
- `analyze_failures.py`: 31/1000, no regressions
- `test_pipeline.py` 2/2, `dryrun_fixture.py` 2/2, synth smoke test 20/20

## Next steps (need Kaggle GPU — can't be measured locally)
1. **Tune priming + DFS**: priming steps/LR (`ttt.prime_proposer`), DFS beam
   width/time (`models_nvarc._propose_turbo`), rerank weight — measure on eval.
2. **Scale synth to 10k+** and fine-tune Qwen3 offline; NVARC used 3.2M samples —
   this is the step that took them from ~16% to 24%.
3. **Debug the 16 missed pure-recolor fails** — colormap probe edge case
   (likely mapping/filter bug in `_color_map_source` / `_colormap_d4_source`).
4. **Object logic for the 647 same-shape structural fails** — beyond
   gravity/mirror: sort-by-size, symmetry completion variants, counting.
5. **Push + run**: commit → GitHub → upload notebook to Kaggle
   (`rahulvats20/arc-agi-2-neuro-symbolic-solver`), attach competition input +
   `arc-agi-2-pkg` dataset + Qwen3 model, run on L4x4, submit.

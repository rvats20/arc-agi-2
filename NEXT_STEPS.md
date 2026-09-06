# Session updates (2026-09-06) — Tracks 1+2

## Track 1: DSL boost (2026-09-06)
- 5 new primitives in `arc_agi2/dsl.py`: `filter_objects_by_size`, `remove_small_objects`, `keep_n_largest`, `fill_holes`, `upscale_with_mode`
- New probe `_filter_holes_source` (10 candidates) + extended `_composition_sources` with new primitives as outer ops
- System prompts updated (`models.py`, `models_nvarc.py`) → now 24 primitives listed
- Synth `PIPELINES` expanded to include new primitives
- **Result: 31/1000 unchanged** — new primitives hit 0/200 on sampled structural tasks (too simple for the 647 structural bucket which needs counting/replication/logic, not just filter+fill). Conclusion: DSL ceiling is ~3-5% as expected; further DSL gains need composition search or code-gen, not more single-op probes.

## Track 2: Synth scale (2026-09-06)
- Scaled `data/synth` from 20 → **2000 synthetic tasks** (`python -m arc_agi2.synth --n 2000 --seed 0`)
- `data/synth_test` still holds 200 (smoke test set)
- Pipelines now include filter/holes primitives, so synthetic distribution covers new capability
- `pkg_dataset/arc_agi2/` synced (18 py files) for Kaggle dataset publish

## Verified (CPU)
- `analyze_failures.py`: **31/1000 = 3.10%** (no regression, but no new solves)
- `test_pipeline.py`: 2/2, `dryrun_fixture.py`: 2/2, synth 200/200 valid
- Notebook rebuilt: `build_notebook.py` → `arc_agi2_solver.ipynb` (16519 B) → synced to `kaggle_push/`

## What changed this session (prior: 2026-09-05)
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
- System prompts (`models.py`, `models_nvarc.py`) list all primitives
- Notebook rebuilt via `build_notebook.py`

## Next steps (need Kaggle GPU — can't be measured locally)
1. **Push + run**: commit done (18b9c38) → push to GitHub → upload notebook to Kaggle
   (`rahulvats20/arc-agi-2-neuro-symbolic-solver`), attach competition input +
   `arc-agi-2-pkg` dataset + Qwen3 model, run on L4x4, submit.
2. **Tune priming + DFS** on GPU: `ttt.prime_proposer` steps/LR, DFS beam width/time, rerank weight — measure on eval (120 tasks).
3. **Scale synth to 10k+** and fine-tune Qwen3 offline; NVARC used 3.2M samples — this is the step that took them from ~16% to 24%.
4. **DSL ceiling**: further single-op primitives unlikely to help (proven 0/200 hit rate). If DSL is pursued, add **program search** (enumerate 2-3 op compositions via DFS) rather than more hand-crafted probes.

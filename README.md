# ARC-AGI-2 Neuro-Symbolic Solver

LLM-proposes + symbolic verifier-checks, targeting **arc-prize-2026-arc-agi-2**.
The library is local-CPU-safe; the VLM (Qwen2.5-VL-7B, 4-bit) runs only on a
Kaggle GPU notebook and is strictly separated from the verifier: the model can
hallucinate freely, only programs that exactly reproduce the train outputs are
kept and used to predict the test input.

## Repo layout
```
arc-agi-2/
  arc_agi2/        # library (imports cleanly on CPU)
    loader.py      # ARC-AGI-2 data loading (configurable ARC_DATA_DIR)
    grid_utils.py  # grid -> ASCII / PIL image (for the VLM)
    dsl.py         # symbolic primitives + SAFE_GLOBALS
    verifier.py    # verify_program / run_program (the symbolic gate)
    submission.py  # submission.json writer + validator (attempt_1/attempt_2)
    models.py      # QwenVL (lazy GPU import, 4-bit, multi-candidate propose)
    mock_vl.py     # CPU-only stand-in for dry-runs (NOT a real LLM)
  data/            # ARC-AGI-2 training/eval/test (downloaded from Kaggle)
  arc_agi2_solver.ipynb   # submission-ready Kaggle notebook (self-contained)
  test_pipeline.py        # CPU self-test + real-eval DSL floor
  dryrun_local.py         # full-loop dry-run with MockVL (no GPU)
  dryrun_fixture.py       # full-loop dry-run on a 2-task fixture
```

## Local CPU commands (no model needed)
```bash
cd /mnt/c/Users/Rahul/arc-agi-2
. .venv/bin/activate
export ARC_DATA_DIR=/mnt/c/Users/Rahul/arc-agi-2/data

python test_pipeline.py     # fixture 2/2 + real eval DSL floor (0/120 expected)
python dryrun_local.py      # full DSL->MockVL->verify->submit loop (CPU)
python dryrun_fixture.py    # proves the accept-path fires (2/2)
```

## Measured status (this session)
- Fixture pipeline: 2/2 solved, submission valid.
- ARC-AGI-2 training set, pure DSL: 24/1000 = 2.40%
  (v1.0 baseline 14, +5 colormap fix, +3 tile primitives, +2 fill_enclosed)
- Real ARC-AGI-2 eval, toy DSL: 0/120 (honest floor; VLM is the scorer).
- Full loop with MockVL: accept-path verified on fixture; 0/120 on real eval
  (real tasks aren't pure transforms — expected).
- Notebook: 19 cells, all compile; Qwen load + verifier + checkpoint + format present.
- LLM system prompt lists all 16 primitives; failure hint includes ASCII diff
  for grids <= 20x20; repair loop picks the best (lowest diff) candidate.

## Run on Kaggle (real score)
1. Create a Kaggle dataset with **Qwen2.5-VL-7B-Instruct** in 4-bit. Easiest:
   download the `unsloth/Qwen2.5-VL-7B-Instruct-4bit` (or any 4-bit GGUF/safetensors
   build) during dev, then `kaggle datasets create -p ./qwen25vl_4bit`.
   The notebook expects it at `/kaggle/input/models/Qwen2.5-VL-7B-Instruct-4bit`.
   Change `QWEN_PATH` in the config cell if your dataset path differs.
2. Enter the competition `arc-prize-2026-arc-agi-2` and create a **notebook**
   (not a script) attached to the competition input + your model dataset.
3. Set the notebook accelerator to **L4x4 (96GB)** so the 7B model (~15GB 4-bit)
   fits and 2x GPU quota is consumed as the competition allows.
4. Upload `arc_agi2_solver.ipynb` as the notebook. The library is inlined, so no
   extra file upload is needed. No internet is used at eval time.
5. Run. Output: `/kaggle/working/submission.json` (validated before write).
6. Submit the notebook (code competition: the notebook itself is the submission).

## Config knobs (top of the notebook config cell)
- `KAGGLE_INPUT`  – competition input mount (verify the slug matches your Kaggle mount).
- `QWEN_PATH`     – 4-bit model dataset path.
- `n_candidates`  – VLM proposals per task (raise for more search, costs time).
- `HARD_LIMIT_S` / `FINALIZE_RESERVE` – 10h hard cap, 15m finalize reserve.

## Scoring note
Each test input scores if EITHER `attempt_1` or `attempt_2` matches ground truth
exactly. We set `attempt_2` = identity fallback so the grid is never empty; a
verified program overrides `attempt_1`. The real leaderboard number only appears
after a Kaggle GPU run — it cannot be measured locally (no GPU, eval outputs withheld).

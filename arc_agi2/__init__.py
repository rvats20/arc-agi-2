"""ARC-AGI-2 neuro-symbolic solver library.

Layout:
  loader.py      - load challenges/solutions for ARC-AGI-2 file naming
  grid_utils.py  - grid <-> ASCII / PIL image rendering (for Qwen2.5-VL)
  dsl.py         - symbolic primitive transforms + brute-force search baseline
  verifier.py    - run a candidate program against train pairs, check exact match
  models.py      - Qwen2.5-VL interface (guarded for Kaggle GPU; lazy imports)
  submission.py  - write submission.json in the official format
  cache.py       - task-shape fingerprint cache (so re-runs are free)
  grid_text.py   - Qwen-style grid-as-text encoding (digits + newlines)
  arc_dataset.py - ArcDataset with D4 + S10 augmentation (VARC/NVARC port)
  selection.py   - score_kgmon / score_full_probmul_3 (VARC/NVARC port)

All CPU-safe functions (loader/grid_utils/dsl/verifier/submission/
cache/grid_text/arc_dataset/selection) run and verify locally on WSL.
models.py imports torch/transformers only inside its functions so it
stays importable on CPU machines.
"""

from .loader import Task, load_task, load_challenges, load_solutions, load_all
from .grid_utils import grid_to_ascii, grid_to_image, render_task_thumbnails
from .dsl import search_solve, PRIMITIVES
from .verifier import verify_program, run_program
from .submission import write_submission, empty_submission, validate_submission

__all__ = [
    "Task", "load_task", "load_challenges", "load_solutions", "load_all",
    "grid_to_ascii", "grid_to_image", "render_task_thumbnails",
    "search_solve", "PRIMITIVES", "verify_program", "run_program",
    "write_submission", "empty_submission", "validate_submission",
]

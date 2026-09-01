"""Qwen2.5-VL interface for ARC-AGI-2 (bundled, offline on Kaggle).

LAZY imports: torch/transformers only imported inside functions, so the rest
of the library imports cleanly on CPU-only machines (WSL, no GPU).

Design (neuro-symbolic):
  1. Render train pairs + test input as images.
  2. Ask the VLM to WRITE a Python `solve(g)` function (g = numpy int array).
  3. propose() returns several candidate source strings (sampled, temp>0).
  4. The NOTEBOOK verifies each candidate with verifier.verify_program against
     the train pairs; the first that reproduces all train outputs exactly is
     used to generate the test prediction. Unverified proposals are discarded.

This keeps the model proposer and the symbolic verifier strictly separated:
the model can hallucinate freely; only verified programs reach the submission.
"""
from __future__ import annotations

import re
import textwrap
from typing import Optional

import numpy as np

SAFE_COLORS_NOTE = (
    "Grid colors are ints 0-9; 0 is the black background. "
    "Grids are rendered as colored cells in the images."
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an ARC-AGI grid-reasoning solver. You receive example input/output
    grids as images and one test input grid. Infer the transformation rule
    and WRITE a Python function `solve(g)` where `g` is a numpy int array
    (rows x cols). Available helpers (already in scope):

      ORIENTATION:
        rotate_cw, rotate_ccw, flip_h, flip_v, transpose
      COLOR:
        invert_colors, color_replace(g, src, dst), keep_color(g, c)
      GEOMETRY:
        crop_nonzero, bounding_box_crop, shift_to_origin, grow_nonzero
        scale_up(g, k), scale_down(g, k)
      TILING (use when output is a self-similar pattern of the input):
        kron_tile(g, k)         - repeat the input k x k
        masked_kron_tile(g, k)  - use input as a binary mask; place a copy
                                  of g at each non-zero cell of the mask
        brickwall_tile(g, k)    - k x k tiling with horizontal-flip on
                                  every odd row of tiles
      REGION:
        find_objects(g, bg)             - list of connected-component masks
        fill_enclosed(g, frame, fill)   - fill zero-regions enclosed by
                                          frame-color (4-connected)
        flood_fill_4(g, (y, x), color)   - 4-connected paint from seed

    You may also write plain numpy / Python code. `solve(g)` must return a
    numpy int array (the output grid). Output ONLY python code starting
    with `def solve(g):` and nothing else. No markdown, no commentary.
    """
).strip()


def _extract_code(text: str) -> str:
    """Pull a `def solve(g):` block out of model output, tolerating markdown."""
    # Strip a leading system/assistant wrapper if present.
    m = re.search(r"def solve\(g\):", text)
    if not m:
        return ""
    start = m.start()
    body = text[start:]
    # If fenced, cut at the closing fence.
    fence = re.search(r"```", body[start - 0:])  # search whole body for a fence after start
    # Simpler: find first ``` after start
    fence_match = re.search(r"```", body)
    if fence_match and fence_match.start() > 0:
        # only treat as fence if it appears after the def
        cand = body[fence_match.start():]
        # ensure the fence is after the def start
        if fence_match.start() > 0:
            body = body[:fence_match.start()]
    return body.strip()


class QwenVL:
    def __init__(self, model_path: str, device: str = "auto",
                 load_in_4bit: bool = True, dtype: str = "bfloat16"):
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                                  BitsAndBytesConfig)
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.dtype, torch.bfloat16)
        kwargs = {"device_map": self.device, "torch_dtype": dt}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, **kwargs
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def _build_messages(self, task, hint: Optional[str] = None):
        from .grid_utils import render_task_thumbnails
        from PIL import Image

        imgs = render_task_thumbnails(task, cell=32)
        content = []
        idx = 0
        for i, pair in enumerate(task.train):
            content.append({"type": "image", "image": imgs[idx]})
            content.append({"type": "text", "text": f"train {i} INPUT"})
            idx += 1
            if "output" in pair:
                content.append({"type": "image", "image": imgs[idx]})
                content.append({"type": "text", "text": f"train {i} OUTPUT"})
                idx += 1
        for j in range(len(task.test)):
            content.append({"type": "image", "image": imgs[idx]})
            content.append({"type": "text", "text": f"test {j} INPUT -> write solve(g)"})
            idx += 1
        content.append({"type": "text", "text": SAFE_COLORS_NOTE})
        if hint:
            content.append({"type": "text", "text":
                "Your previous attempt failed. Correction hint: " + hint})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def propose(self, task, n_candidates: int = 4, max_new_tokens: int = 1024,
                temperature: float = 0.7, hint: Optional[str] = None) -> list[str]:
        """Return a list of candidate `solve(g)` source strings (unverified)."""
        self._load()
        messages = self._build_messages(task, hint=hint)
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [c["image"] for c in messages[1]["content"] if c["type"] == "image"]
        inputs = self._processor(
            text=prompt, images=images, return_tensors="pt"
        ).to(self._model.device)
        out = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature,
            top_p=0.95, num_return_sequences=n_candidates,
        )
        decoded = self._processor.batch_decode(out, skip_special_tokens=True)
        cands = []
        for d in decoded:
            src = _extract_code(d)
            if src:
                cands.append(src)
        return cands

    def propose_verified(self, task, n_candidates: int = 4) -> Optional[str]:
        """Return the first proposed program that verifies against train pairs,
        or None. (Kept here for convenience; the notebook usually does this to
        interleave checkpointing.)"""
        from .verifier import verify_program
        for src in self.propose(task, n_candidates=n_candidates):
            if verify_program(src, task):
                return src
        return None


def looks_llm_hopeless(task, src: str | None) -> bool:
    """Heuristic: a task that the DSL can't solve and has more than ~3 train
    pairs and very dense inputs is likely too hard for the LLM too (the LLM
    would burn 10+ minutes on it). Skip the LLM branch and let attempt_2
    (identity) be the answer.

    This is a budget safeguard, not a quality signal. False negatives cost
    us a possible solve; false positives save a lot of Kaggle time.
    """
    n_pairs = len(task.train)
    if n_pairs < 4:
        return False
    # Average density across train inputs
    densities = []
    for pair in task.train:
        inp = np.asarray(pair["input"], dtype=int)
        densities.append(float((inp != 0).sum()) / max(inp.size, 1))
    avg_density = sum(densities) / max(len(densities), 1)
    return avg_density > 0.7


def repair_loop(proposer, task, n_rounds: int = 3, n_candidates: int = 4,
                skip_if_hopeless: bool = True) -> Optional[str]:
    """Neuro-symbolic REPAIR: propose candidates, verify; if the best one
    fails, feed the failing train pair back to the proposer as a concrete
    correction hint (with ASCII diff) and try again. Works with ANY proposer
    exposing propose(task, hint=...) + the shared verifier. Returns a
    verified source or None.

    On Kaggle `proposer` is QwenVL (sees the failure as a natural-language
    hint). Locally we exercise the SAME control flow with MockVL so the
    loop is proven.

    If `skip_if_hopeless` is True (default), tasks that look too dense AND
    have many train pairs are skipped — the LLM would burn 10+ minutes on
    them and likely fail. Set False to never skip.
    """
    from .verifier import verify_program, run_program
    import numpy as np

    if skip_if_hopeless and looks_llm_hopeless(task, None):
        return None

    hint: Optional[str] = None
    best_hint: Optional[str] = None
    for _ in range(n_rounds):
        cands = proposer.propose(task, n_candidates=n_candidates, hint=hint)
        for src in cands:
            if src and verify_program(src, task):
                return src
        # Pick the candidate that's CLOSEST to correct, not just the first
        # — feed that one's failure as the next hint so the LLM can see
        # what was almost-right and adjust.
        if cands:
            scored = []
            for src in cands:
                try:
                    diffs = []
                    for pair in task.train:
                        pred = np.array(run_program(src, pair["input"]), dtype=int)
                        gold = np.array(pair["output"], dtype=int)
                        if pred.shape != gold.shape:
                            diffs.append(pred.size + 1)  # huge penalty
                        else:
                            diffs.append(int((pred != gold).sum()))
                    scored.append((sum(diffs), src))
                except Exception:
                    scored.append((10**9, src))
            scored.sort(key=lambda t: t[0])
            best_src = scored[0][1]
            best_hint = _failure_hint(task, best_src)
            hint = best_hint
    return None


def _failure_hint(task, src: str) -> str:
    """Describe, in plain text, how the candidate failed on the first train
    pair, so a language-model proposer can correct itself.

    The hint is intentionally concrete: it names the failing train pair
    index, the predicted vs expected shape, the number of differing cells,
    and (for small grids) the ASCII diff so the LLM can SEE what's wrong.
    """
    from .verifier import run_program
    import numpy as np
    # Use the FIRST pair that fails (not necessarily pair 0)
    for pi, pair in enumerate(task.train):
        try:
            pred = np.array(run_program(src, pair["input"]), dtype=int)
        except Exception as e:
            return (f"Pair {pi}: your program raised {type(e).__name__}: {e}. "
                    f"Fix the code; remember the available helpers listed in "
                    f"the system prompt.")
        gold = np.array(pair["output"], dtype=int)
        if pred.shape != gold.shape:
            return (f"Pair {pi}: predicted shape {tuple(pred.shape)} but "
                    f"expected {tuple(gold.shape)}. Look at the size ratio "
                    f"to decide whether to use scale_up(k), kron_tile(k), "
                    f"masked_kron_tile(k), or brickwall_tile(k).")
        if not np.array_equal(pred, gold):
            diff_mask = pred != gold
            n_diff = int(diff_mask.sum())
            # Find the bounding box of differences
            ys, xs = np.where(diff_mask)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            hint = (f"Pair {pi}: {n_diff}/{gold.size} cells wrong. "
                    f"Diffs are in rows {y0}..{y1}, cols {x0}..{x1}. ")
            if gold.size <= 400:
                # ASCII diff so the LLM can spot the pattern at a glance
                from .grid_utils import grid_to_ascii
                hint += ("\n  expected:\n" + grid_to_ascii(gold.tolist())
                         + "\n  got:\n" + grid_to_ascii(pred.tolist()))
            return hint
    return "All train pairs matched."  # shouldn't reach here normally

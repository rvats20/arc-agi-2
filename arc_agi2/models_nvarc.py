"""Qwen3-4B proposer for ARC-AGI-2 (port from the NVARC reference).

Uses the publicly-available Qwen3-4B fine-tuned for ARC grids
(`sorokin/qwen3_4b_grids15_sft139`) with the chat-template-based
grid-as-text encoding. Proposes solve(g) Python source strings, and
optionally runs a Turbo DFS beam search if `turbo_dfs=True` is passed
to propose().

The Turbo DFS implementation matches the reference cell 4 of
`arc2-nvarc-v1.ipynb`. The non-turbo path uses sampling generate()
which is faster but lower-quality.

Heavy on torch/transformers. Lazy-imports everything so the rest of
arc_agi2 stays importable on CPU.
"""
from __future__ import annotations
import re
import textwrap
import time
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore[assignment]


SYSTEM_PROMPT = textwrap.dedent("""
You are an ARC-AGI grid-reasoning solver. You receive example input/output
grids as digit text and a test input grid. WRITE a Python function
`solve(g)` where `g` is a numpy int array. Available helpers (already in scope):
  np, rotate_cw, rotate_ccw, flip_h, flip_v, transpose, invert_colors,
  crop_nonzero, grow_nonzero, kron_tile(g, k), masked_kron_tile(g, k),
  brickwall_tile(g, k), shift_to_origin, color_replace, keep_color,
  find_objects, fill_enclosed, flood_fill_4, scale_up, scale_down.
You may also write plain numpy / Python code. `solve(g)` must return a
numpy int array. Output ONLY python code starting with `def solve(g):`
and nothing else. No markdown, no commentary.
""").strip()


def require_torch() -> None:
    if not _TORCH_OK:
        raise RuntimeError("models_nvarc needs torch")


def _extract_code(text: str) -> str:
    """Pull a `def solve(g):` block out of model output, tolerating markdown."""
    m = re.search(r"def solve\(g\):", text)
    if not m:
        return ""
    body = text[m.start():]
    fence = re.search(r"```", body)
    if fence and fence.start() > 0:
        body = body[:fence.start()]
    return body.strip()


def _grid_to_text(grid) -> str:
    """2D grid -> 'row1\\nrow2\\n...'. One digit per cell, newline between rows."""
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()


def _text_to_grid(text: str) -> Optional[np.ndarray]:
    """Inverse of _grid_to_text. Stops at the first line that's all <|im_end|>
    or empty. Returns None if the text doesn't form a valid grid."""
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return None
    rows = []
    for ln in lines:
        if "<|im_end|>" in ln:
            ln = ln.split("<|im_end|>")[0]
        if not ln:
            continue
        try:
            row = [int(c) for c in ln if c.isdigit()]
        except ValueError:
            return None
        if row:
            rows.append(row)
    if not rows:
        return None
    w = len(rows[0])
    if not all(len(r) == w for r in rows):
        return None
    if not (1 <= len(rows) <= 30 and 1 <= w <= 30):
        return None
    arr = np.array(rows, dtype=int)
    if arr.min() < 0 or arr.max() > 9:
        return None
    return arr


def _build_messages(train_pairs, test_input, hint: Optional[str] = None) -> List[Dict[str, str]]:
    """Build a chat-template prompt for the Qwen3-4B model. Each train pair
    is shown as a user/assistant dialog; the test input is a fresh user
    turn with no assistant reply (so the model generates the output)."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for p in train_pairs:
        msgs.append({"role": "user", "content": _grid_to_text(p["input"])})
        msgs.append({"role": "assistant", "content": _grid_to_text(p["output"])})
    msgs.append({"role": "user", "content": _grid_to_text(test_input)
                + ("\n\n# Correction: " + hint if hint else "")})
    return msgs


class Qwen3GridProposer:
    """Qwen3-4B (fine-tuned for ARC grids) proposer. Mimics the reference
    notebook's QwenVL API so existing code can use it as a drop-in."""

    def __init__(self, model_path: str, device: str = "cuda",
                 load_in_4bit: bool = True, max_seq_len: int = 8192,
                 turbo_dfs: bool = True) -> None:
        require_torch()
        self.model_path = model_path
        self.device = device
        self.turbo_dfs = turbo_dfs
        self._tokenizer = None
        self._model = None
        self._arc_token_ids = None
        self.max_seq_len = max_seq_len
        # Lazy import on first use (huge memory hit otherwise)
        self._load(load_in_4bit=load_in_4bit)

    def _load(self, load_in_4bit: bool = True) -> None:
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  BitsAndBytesConfig)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        if load_in_4bit:
            try:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception as e:
                # bitsandbytes may be unavailable (no internet, missing wheel).
                # Fall back to bf16 — uses ~16GB instead of ~4GB but works
                # on any T4/L4 without pip install.
                print(f"[Qwen3] bitsandbytes unavailable ({e}); using bf16")
                kwargs.pop("quantization_config", None)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **kwargs
        ).eval()
        # Pre-extract just the 13 ARC token IDs for fast logit extraction
        arc_token_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]
        self._arc_token_ids = torch.tensor(arc_token_ids, dtype=torch.long,
                                            device=self._model.device)
        print(f"[Qwen3] loaded {self.model_path} (4bit={load_in_4bit})")

    def propose(self, task, n_candidates: int = 4, max_new_tokens: int = 1024,
                temperature: float = 0.7, hint: Optional[str] = None) -> List[str]:
        """Generate n_candidates solve() proposals for the first test input.
        If turbo_dfs=True (default), uses the 13-token beam search; otherwise
        uses sampling generate() (faster but lower quality)."""
        from .verifier import run_program
        train_pairs = [{"input": np.asarray(p["input"], dtype=int),
                        "output": np.asarray(p["output"], dtype=int)}
                       for p in task.train]
        test_input = np.asarray(task.test[0]["input"], dtype=int)
        msgs = _build_messages(train_pairs, test_input, hint=hint)
        prompt = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if self.turbo_dfs:
            return self._propose_turbo(prompt, n_candidates, max_new_tokens)
        return self._propose_sample(prompt, n_candidates, max_new_tokens, temperature)

    def _propose_sample(self, prompt: str, n_candidates: int,
                         max_new_tokens: int, temperature: float) -> List[str]:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature,
            top_p=0.95, num_return_sequences=n_candidates,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.batch_decode(out, skip_special_tokens=False)
        cands = []
        for d in decoded:
            src = _extract_code(d)
            if src:
                cands.append(src)
        return cands

    def _propose_turbo(self, prompt: str, n_candidates: int,
                        max_new_tokens: int) -> List[str]:
        """Turbo DFS beam search over the 13 ARC tokens.

        Ported from the reference notebook. Stays on GPU; uses KV cache;
        only extracts logits for the 13 ARC tokens (much faster than
        sampling the full vocabulary).
        """
        from .verifier import run_program
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, use_cache=True, return_dict=True)
        prefix_tokens = inputs["input_ids"][0].tolist()
        # max_new_tokens from the formatter: 30x30 grid = ~962 tokens
        # but capped to keep latency sane
        max_new = min(max_new_tokens, 962)
        max_score = -np.log(0.2)  # 1.609
        dfs_window = 540.0
        suffixes = self._turbo_dfs(
            self._model, outputs.logits[:, -1],
            max_new_tokens=max_new, max_score=max_score,
            scores=[0.0] * 1, pos=len(prefix_tokens),
            cache=outputs.past_key_values,
            start_time=time.time(),
            end_time=time.time() + dfs_window,
            dfs_window=dfs_window,
        )
        # Build candidates: each suffix is a sequence of token IDs
        cands = []
        for _batch_id, beams in suffixes.items():
            for _score, suffix in beams[:n_candidates]:
                # Decode the suffix tokens (which are ONLY the generated
                # portion) to text
                text = self._tokenizer.decode(suffix, skip_special_tokens=False)
                # Convert to grid, then wrap in a solve() that returns it
                grid = _text_to_grid(text)
                if grid is None:
                    continue
                # We can't easily wrap this as a solve() because the source
                # would be just a constant — but verify_program will accept it
                src = f"def solve(g):\n    return np.array({grid.tolist()}, dtype=int)\n"
                cands.append(src)
        return cands

    def _turbo_dfs(self, model, logits, max_new_tokens, max_score,
                    scores, pos, cache, start_time, end_time, dfs_window,
                    n=1) -> Dict[int, List[Tuple[float, List[int]]]]:
        """Recursive beam search over the 13-token ARC vocabulary. Stops when
        out of time or no more expansions. Returns {batch_id: [(score, [tok, tok, ...]), ...]}."""
        logits_f = logits.float()
        arc_logits = logits_f.index_select(-1, self._arc_token_ids)
        nll = (torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
                + torch.logsumexp(logits_f, dim=-1, keepdim=True)
                - arc_logits).cpu()
        suffixes: Dict[int, List[Tuple[float, List[int]]]] = {}
        candidates: Dict[int, List[Tuple[float, int]]] = {}
        for i in range(n):
            candidates[i] = []
            for tok_idx, t in enumerate(self._arc_token_ids.tolist()):
                score = nll[i, tok_idx].item()
                if score < max_score:
                    if t == 15:  # <|im_end|>
                        suffixes.setdefault(i, []).append((score, [t]))
                    elif max_new_tokens > 1:
                        candidates[i].append((score, t))
        for i in range(n):
            candidates[i] = sorted(candidates[i], key=lambda x: x[0])
        while time.time() - start_time < dfs_window and time.time() < end_time:
            batch_tokens: List[int] = []
            batch_scores: List[float] = []
            alive = 0
            for i in range(n):
                if not candidates[i]:
                    batch_tokens.append(0)
                    batch_scores.append(1000.0)
                else:
                    score, t = candidates[i].pop(0)
                    batch_tokens.append(t)
                    batch_scores.append(score)
                    alive += 1
            if alive == 0:
                break
            outputs = model(
                input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
                position_ids=torch.full((n, 1), pos, device=model.device),
                past_key_values=cache, return_dict=True, use_cache=True,
            )
            next_suf = self._turbo_dfs(
                model, outputs.logits[:, -1],
                max_new_tokens=max_new_tokens - 1, max_score=max_score,
                scores=batch_scores, pos=pos + 1,
                cache=outputs.past_key_values,
                start_time=start_time, end_time=end_time,
                dfs_window=dfs_window, n=n,
            )
            for bid, beams in next_suf.items():
                for score, toks in beams:
                    toks = [batch_tokens[bid]] + toks
                    suffixes.setdefault(bid, []).append((score, toks))
        return suffixes

"""Qwen-style grid-as-text encoding for ARC-AGI-2.

Each grid is a flat string of digits (0-9) joined by '\\n' between rows.
Qwen3-4B fine-tuned for ARC (sorokin/qwen3_4b_grids15_sft139) uses this
format with a 13-token vocabulary: digits 0-9, newline, <|im_start|>,
<|im_end|>, pad.

This lets us:
  * Compute NLL on a tiny vocabulary (fast logit extraction)
  * Decode beam-search outputs back to grids via convert_tokens_to_array
  * Augment with D4 (rotations + reflections) and S10 (color permutations)
    by applying numpy ops and re-stringifying.

Ported from the public NVARC + VARC reference notebooks (LB 33.89 lineage).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional


# The 13-token ARC vocabulary.  Keys are the Qwen tokenizer strings;
# values are the token IDs assigned in the fine-tuned model.
ARC_VOCAB = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "\n": 10,
    "<|im_start|>": 11,
    "<|im_end|>": 15,
}
# Token IDs that are actual content (digits + newline)
ARC_CONTENT_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ARC_END_OF_GRID = 15  # <|im_end|>


def convert_grid_to_string(grid) -> str:
    """Flatten a 2D grid to "row1\\nrow2\\n..." with one digit per cell."""
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()


def is_valid_solution(guess) -> bool:
    """A grid is valid if it's a 2D int array with shape in (0, 30] x (0, 30]."""
    if not isinstance(guess, np.ndarray) or guess.ndim != 2:
        return False
    return all(0 < x <= 30 for x in guess.shape)


def hashable(guess) -> tuple:
    """Make a grid hashable so we can use it as a dict key / count votes."""
    return tuple(map(tuple, guess))


def permute_mod(a, permutation_descriptor, invert=False):
    """Apply or invert a color permutation. descriptor is a string of 10 digits
    giving the new color for each original color (0..9 -> permutation[c])."""
    permutation = [int(i) for i in permutation_descriptor if str(i).isdigit()]
    assert sorted(permutation) == list(range(10))
    a = np.asarray(a)
    if invert:
        permutation = np.argsort(permutation)
    a = np.asarray(permutation)[a]
    return a


def random_perm_descriptor() -> str:
    """Generate a random 10-digit permutation descriptor (e.g. 'permute4965803712')."""
    p = np.random.permutation(10).tolist()
    return "permute" + "".join(map(str, p))


def convert_tokens_to_array(tokens, limit_rows: int = 30) -> Optional[np.ndarray]:
    """Decode a list of token IDs (digits + newlines) back to a 2D numpy grid.
    Stops at the first <|im_end|> (15) or end of list. Returns None if the
    decoded grid isn't a valid ARC solution.
    """
    if len(tokens) < 2:
        return None
    # Walk until <|im_end|> or end
    digits = []
    for t in tokens:
        if t == ARC_END_OF_GRID:
            break
        if t in range(10):
            digits.append(t)
        elif t == 10:  # \n
            digits.append(-1)
    # Split on -1 (newline) into rows; drop empty rows
    rows = []
    cur = []
    for d in digits:
        if d == -1:
            if cur:
                rows.append(cur)
                cur = []
        else:
            cur.append(d)
    if cur:
        rows.append(cur)
    if not rows:
        return None
    rows = rows[:limit_rows]
    # All rows must be equal length for a valid grid
    width = len(rows[0])
    if not all(len(r) == width for r in rows):
        return None
    if not (1 <= len(rows) <= 30 and 1 <= width <= 30):
        return None
    arr = np.array(rows, dtype=int)
    if arr.min() < 0 or arr.max() > 9:
        return None
    return arr

"""Symbolic DSL + brute-force baseline search for ARC-AGI-2.

Design goals:
  * A program is a source string defining `solve(g)` where g is an int ndarray.
  * SAFE_GLOBALS restricts what a program (LLM-proposed) is allowed to do.
  * PRIMITIVES is a library of grid ops usable inside `solve`.
  * search_solve() brute-forces a small space of single-primitive programs to
    give an honest, MEASURED DSL floor (the spec notes pure DSL ~2%).

NOTE: this is intentionally a warm-up. The real wins come from the LLM
proposing richer `solve` bodies; the verifier accepts any valid Python that
runs in SAFE_GLOBALS.
"""
from __future__ import annotations

import numpy as np

# --- safe execution namespace -------------------------------------------------
SAFE_GLOBALS = {
    "np": np,
    "ndarray": np.ndarray,
    "__builtins__": {},  # no open/exec/eval etc.
}

# --- primitive grid ops (callable inside solve via np / these helpers) --------
def rotate_cw(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=-1)

def rotate_ccw(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=1)

def flip_h(g: np.ndarray) -> np.ndarray:
    return np.fliplr(g)

def flip_v(g: np.ndarray) -> np.ndarray:
    return np.flipud(g)

def transpose(g: np.ndarray) -> np.ndarray:
    return g.T

def invert_colors(g: np.ndarray) -> np.ndarray:
    # 0<->9, 1<->8, ... mirror around 4.5
    return (9 - g) % 10

def crop_nonzero(g: np.ndarray) -> np.ndarray:
    mask = g != 0
    if not mask.any():
        return g
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

def grow_nonzero(g: np.ndarray, pad: int = 1) -> np.ndarray:
    return np.pad(g, pad, mode="constant", constant_values=0)

def scale_up(g: np.ndarray, k: int = 2) -> np.ndarray:
    """Up-sample by repeating each cell k x k (nearest-neighbor scale)."""
    if k <= 1:
        return g
    h, w = g.shape
    out = np.zeros((h * k, w * k), dtype=g.dtype)
    for y in range(h):
        for x in range(w):
            out[y * k:(y + 1) * k, x * k:(x + 1) * k] = g[y, x]
    return out

def scale_down(g: np.ndarray, k: int = 2) -> np.ndarray:
    """Down-sample by max-pooling blocks of k x k (keeps any non-zero)."""
    if k <= 1:
        return g
    h, w = g.shape
    hh, ww = h // k, w // k
    out = np.zeros((hh, ww), dtype=g.dtype)
    for y in range(hh):
        for x in range(ww):
            block = g[y * k:(y + 1) * k, x * k:(x + 1) * k]
            uniq = block[block != 0]
            out[y, x] = int(uniq[0]) if uniq.size else 0
    return out

def reflect_diag(g: np.ndarray) -> np.ndarray:
    return np.transpose(g)

def bounding_box_crop(g: np.ndarray) -> np.ndarray:
    return crop_nonzero(g)

def color_replace(g: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = g.copy()
    out[out == src] = dst
    return out

def keep_color(g: np.ndarray, c: int) -> np.ndarray:
    """Mask to only color c (everything else -> 0)."""
    out = np.zeros_like(g)
    out[g == c] = c
    return out

PRIMITIVES = {
    "rotate_cw": rotate_cw,
    "rotate_ccw": rotate_ccw,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "invert_colors": invert_colors,
    "crop_nonzero": crop_nonzero,
    "grow_nonzero": grow_nonzero,
    "scale_up": scale_up,
    "scale_down": scale_down,
    "bounding_box_crop": bounding_box_crop,
    "color_replace": color_replace,
    "keep_color": keep_color,
}

# Register primitives into the safe namespace so solve() can call them directly.
for _name, _fn in PRIMITIVES.items():
    SAFE_GLOBALS[_name] = _fn


def _single_sources() -> list[str]:
    srcs = []
    for name in PRIMITIVES:
        srcs.append(f"def solve(g):\n    return {name}(g)\n")
    # common fixed-arg variants
    srcs.append("def solve(g):\n    return scale_up(g, 2)\n")
    srcs.append("def solve(g):\n    return scale_up(g, 3)\n")
    srcs.append("def solve(g):\n    return scale_down(g, 2)\n")
    srcs.append("def solve(g):\n    return rotate_cw(rotate_cw(g))\n")
    srcs.append("def solve(g):\n    return rotate_cw(rotate_cw(rotate_cw(g)))\n")
    return srcs


def _composition_sources() -> list[str]:
    """Cheap 2-primitive compositions (outer(inner(g)))."""
    outs = ["rotate_cw", "rotate_ccw", "flip_h", "flip_v", "transpose",
            "crop_nonzero", "invert_colors", "scale_up", "scale_down"]
    srcs = []
    for o in outs:
        for i in ["rotate_cw", "rotate_ccw", "flip_h", "flip_v", "transpose",
                  "crop_nonzero", "invert_colors"]:
            if o == i:
                continue
            srcs.append(f"def solve(g):\n    return {o}({i}(g))\n")
    # a few scale->orientations
    srcs.append("def solve(g):\n    return rotate_cw(scale_up(g, 2))\n")
    srcs.append("def solve(g):\n    return crop_nonzero(scale_up(g, 2))\n")
    return srcs


def _color_map_source(task) -> str | None:
    """If every train pair has a consistent per-color map input->output
    (shapes can differ across pairs), return a solve() that applies it.

    Two cells only get a useful map when the mapping is non-trivial (size>=2)
    and uses at least one non-zero source. Identity-only maps are skipped.
    Unmapped source colors map to 0 (background).
    """
    mapping: dict[int, int] = {}
    for pair in task.train:
        inp = np.array(pair["input"], dtype=int)
        out = np.array(pair["output"], dtype=int)
        # Each pair must preserve shape (color map cannot change size)
        if inp.shape != out.shape:
            return None
        for a, b in zip(inp.flat, out.flat):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    # Need a meaningful (non-identity) map with at least one non-zero source
    if not mapping or len(mapping) < 2:
        return None
    if all(k == v for k, v in mapping.items()):
        return None
    if all(k == 0 for k in mapping.keys()):
        return None
    # Build lookup: 0..9 -> mapped value or 0 if unmapped
    f = [0] * 10
    for k, v in mapping.items():
        f[int(k)] = int(v)
    return (f"def solve(g):\n"
            f"    _f = np.array({f}, dtype=np.int64)\n"
            f"    return _f[g]\n")


def synthesize(task, max_compose: bool = True, verbose: bool = False) -> str | None:
    """Bounded program synthesizer over the primitive library.

    Tries (in order, cheapest first): single primitives -> color-map ->
    2-primitive compositions. Returns the first source that verifies against all
    train pairs, or None. This is the symbolic floor; the LLM extends beyond it.
    """
    from .verifier import verify_program
    for src in _single_sources():
        if verify_program(src, task):
            if verbose:
                print("  verified (single):", src.strip().splitlines()[0])
            return src
    cm = _color_map_source(task)
    if cm and verify_program(cm, task):
        if verbose:
            print("  verified (colormap)")
        return cm
    if max_compose:
        for src in _composition_sources():
            if verify_program(src, task):
                if verbose:
                    print("  verified (compose):", src.strip().splitlines()[0])
                return src
    return None


def search_solve(task, verbose: bool = False) -> str | None:
    """Backward-compatible alias for synthesize()."""
    return synthesize(task, verbose=verbose)


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

# --- higher-level shape-aware primitives -------------------------------------

def kron_tile(g: np.ndarray, k: int = 2) -> np.ndarray:
    """Kronecker-style tiling: repeat the entire grid k x k times.

    Equivalent to np.kron(g, np.ones((k, k))) but faster.
    """
    if k <= 1:
        return g
    h, w = g.shape
    out = np.zeros((h * k, w * k), dtype=g.dtype)
    for y in range(h):
        for x in range(w):
            out[y * k:(y + 1) * k, x * k:(x + 1) * k] = g[y, x]
    return out

def masked_kron_tile(g: np.ndarray, k: int = 2) -> np.ndarray:
    """Use the input as a binary mask; place a copy of the input at every
    non-zero cell position in the output, leave zero cells as zero.

    Output is (h*k, w*k). For each (i,j) where g[i,j] != 0, the k x k tile
    at output position (i, j) is filled with g itself; else it's zero.

    This is the "self-similar placement" family: 00576224-style kronecker
    but selective (only where the mask cell is on).
    """
    if k <= 1:
        return g
    h, w = g.shape
    out = np.zeros((h * k, w * k), dtype=g.dtype)
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0:
                out[i * k:(i + 1) * k, j * k:(j + 1) * k] = g
    return out

def brickwall_tile(g: np.ndarray, k: int = 2) -> np.ndarray:
    """Tile the input k x k times, but flip-h every odd row of tiles.

    Output is (h*k, w*k). Even tile rows = g, odd tile rows = flip_h(g).
    """
    if k <= 1:
        return g
    h, w = g.shape
    out = np.zeros((h * k, w * k), dtype=g.dtype)
    g_flip = np.fliplr(g)
    for ti in range(k):
        row = g if ti % 2 == 0 else g_flip
        for tj in range(k):
            out[ti * h:(ti + 1) * h, tj * w:(tj + 1) * w] = row
    return out

def flood_fill_4(g: np.ndarray, seed: tuple[int, int], color: int) -> np.ndarray:
    """4-connected flood fill from (y, x) — replaces reachable cells of
    seed value with `color`. Standard paint-bucket.
    """
    h, w = g.shape
    out = g.copy()
    sy, sx = seed
    if not (0 <= sy < h and 0 <= sx < w):
        return out
    target = out[sy, sx]
    if target == color:
        return out
    stack = [(sy, sx)]
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if out[y, x] != target:
            continue
        out[y, x] = color
        stack.append((y + 1, x))
        stack.append((y - 1, x))
        stack.append((y, x + 1))
        stack.append((y, x - 1))
    return out

def fill_enclosed(g: np.ndarray, frame_color: int, fill_color: int) -> np.ndarray:
    """Find regions of 0-cells that are completely surrounded by frame_color
    (4-connected enclosure) and recolor them to fill_color. Cells touching
    the grid border are considered outside and left as 0.

    Implementation: paint the OUTSIDE 0-region (border-reachable zeros) as a
    sentinel, then recolor the remaining zeros (enclosed) to fill_color.
    """
    h, w = g.shape
    out = g.copy()
    SENTINEL = -1
    stack = []
    for x in range(w):
        if out[0, x] == 0:
            stack.append((0, x))
        if out[h - 1, x] == 0:
            stack.append((h - 1, x))
    for y in range(h):
        if out[y, 0] == 0:
            stack.append((y, 0))
        if out[y, w - 1] == 0:
            stack.append((y, w - 1))
    seen = set(stack)
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if out[y, x] != 0:
            continue
        out[y, x] = SENTINEL
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and out[ny, nx] == 0:
                seen.add((ny, nx))
                stack.append((ny, nx))
    # Now fill any remaining 0s (the enclosed ones) with fill_color
    out[out == 0] = fill_color
    # Restore the sentinels to 0
    out[out == SENTINEL] = 0
    return out

def find_objects(g: np.ndarray, bg: int = 0) -> list[dict]:
    """Return a list of connected-component records:
    { 'mask': bool[H,W], 'color': int (the dominant non-bg color), 'n': int }.
    Uses 4-connectivity. Background-color cells inside a shape (holes) are
    NOT considered separate objects.
    """
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objs = []
    for y in range(h):
        for x in range(w):
            if g[y, x] == bg or visited[y, x]:
                continue
            mask = np.zeros((h, w), dtype=bool)
            stack = [(y, x)]
            color = g[y, x]
            n = 0
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or g[cy, cx] == bg:
                    continue
                visited[cy, cx] = True
                mask[cy, cx] = True
                n += 1
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
            objs.append({"mask": mask, "color": color, "n": n})
    return objs

def shift_to_origin(g: np.ndarray, bg: int = 0) -> np.ndarray:
    """Translate the non-background cells so the bounding box starts at (0, 0).
    Cells that fall outside the original grid are dropped; newly-vacated
    cells become bg.
    """
    mask = g != bg
    if not mask.any():
        return g
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    y0, x0 = int(rows.min()), int(cols.min())
    h, w = g.shape
    out = np.full((h, w), bg, dtype=g.dtype)
    # region from (y0, x0) onwards, copied into (0, 0)
    rh = h - y0
    rw = w - x0
    out[:rh, :rw] = g[y0:, x0:]
    return out

def shift_object(g: np.ndarray, dy: int, dx: int, bg: int = 0) -> np.ndarray:
    """Translate the non-background cells by (dy, dx). Cells that fall off
    the grid are dropped; newly-vacated cells become bg.

    dy > 0 = down, dx > 0 = right. Works on the WHOLE object (bounding box
    of non-bg cells), not per-pixel.
    """
    h, w = g.shape
    out = np.full((h, w), bg, dtype=g.dtype)
    mask = g != bg
    if not mask.any():
        return g
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] = g[y, x]
    return out

def find_objects(g: np.ndarray, bg: int = 0) -> list[dict]:
    """Return a list of connected-component records:
    { 'mask': bool[H,W], 'color': int (the dominant non-bg color), 'n': int }.
    Uses 4-connectivity. Background-color cells inside a shape (holes) are
    NOT considered separate objects.
    """
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objs = []
    for y in range(h):
        for x in range(w):
            if g[y, x] == bg or visited[y, x]:
                continue
            mask = np.zeros((h, w), dtype=bool)
            stack = [(y, x)]
            color = g[y, x]
            n = 0
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or g[cy, cx] == bg:
                    continue
                visited[cy, cx] = True
                mask[cy, cx] = True
                n += 1
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
            objs.append({"mask": mask, "color": color, "n": n})
    return objs

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
    "kron_tile": kron_tile,
    "masked_kron_tile": masked_kron_tile,
    "brickwall_tile": brickwall_tile,
    "shift_to_origin": shift_to_origin,
    "shift_object": shift_object,
    "fill_enclosed": fill_enclosed,
    "flood_fill_4": flood_fill_4,
    "find_objects": find_objects,
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
    # parameterized tile variants
    for k in (2, 3, 4):
        srcs.append(f"def solve(g):\n    return kron_tile(g, {k})\n")
        srcs.append(f"def solve(g):\n    return masked_kron_tile(g, {k})\n")
        srcs.append(f"def solve(g):\n    return brickwall_tile(g, {k})\n")
    return srcs


def _composition_sources() -> list[str]:
    """Cheap 2-primitive compositions (outer(inner(g)))."""
    outs = ["rotate_cw", "rotate_ccw", "flip_h", "flip_v", "transpose",
            "crop_nonzero", "invert_colors", "scale_up", "scale_down",
            "kron_tile", "masked_kron_tile", "brickwall_tile", "shift_to_origin"]
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
    # masked_kron with orientation
    for k in (2, 3):
        srcs.append(f"def solve(g):\n    return rotate_cw(masked_kron_tile(g, {k}))\n")
        srcs.append(f"def solve(g):\n    return flip_h(masked_kron_tile(g, {k}))\n")
    return srcs


def _shift_object_source(task) -> str | None:
    """If the transformation is 'shift the (single) non-bg object by (dy, dx)',
    infer dy/dx from the first train pair and emit a solve().

    Heuristic: compare the non-zero cell sets of input and output; the
    (dy, dx) that maps the most input cells into output cells (preserving
    color) is the shift. Works on same-shape tasks only.
    """
    from .verifier import verify_program
    pair = task.train[0]
    inp = np.array(pair["input"], dtype=int)
    out = np.array(pair["output"], dtype=int)
    if inp.shape != out.shape:
        return None
    # Skip if input is blank
    if not (inp != 0).any():
        return None
    h, w = inp.shape
    in_nz = (inp != 0)
    out_nz = (out != 0)
    if in_nz.sum() != out_nz.sum():
        # Object size changed — can't be a pure shift
        return None
    # Try every (dy, dx) in a small window; pick the one that maps the most
    # input cells to output cells of the SAME color
    best = None
    best_score = -1
    for dy in range(-h + 1, h):
        for dx in range(-w + 1, w):
            score = 0
            for y in range(h):
                for x in range(w):
                    if not in_nz[y, x]:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and out[ny, nx] == inp[y, x]:
                        score += 1
            if score > best_score:
                best_score = score
                best = (dy, dx)
    if best is None or best_score != in_nz.sum():
        # No perfect shift found
        return None
    dy, dx = best
    # Try all top shifts (in case of ties) until one verifies
    candidates = []
    for dy_ in range(-h + 1, h):
        for dx_ in range(-w + 1, w):
            score = 0
            for y in range(h):
                for x in range(w):
                    if not in_nz[y, x]:
                        continue
                    ny, nx = y + dy_, x + dx_
                    if 0 <= ny < h and 0 <= nx < w and out[ny, nx] == inp[y, x]:
                        score += 1
            if score == in_nz.sum():
                candidates.append((dy_, dx_))
    for dy_, dx_ in candidates:
        src = f"def solve(g):\n    return shift_object(g, {dy_}, {dx_})\n"
        if verify_program(src, task):
            return src
    return None


def _fill_enclosed_source(task) -> str | None:
    """If the task is 'fill enclosed regions of a frame_color with a fill_color',
    infer (frame_color, fill_color) from the first train pair and emit a solve().

    Heuristic: find the unique non-zero color whose count goes UP between input
    and output (the fill color). The frame color is the most common non-zero
    color in the input (typically the enclosing shape).
    """
    from .verifier import verify_program
    pair = task.train[0]
    inp = np.array(pair["input"], dtype=int)
    out = np.array(pair["output"], dtype=int)
    if inp.shape != out.shape:
        return None
    in_counts = np.bincount(inp.ravel(), minlength=10)
    out_counts = np.bincount(out.ravel(), minlength=10)
    diff = out_counts - in_counts
    # fill_color: the color with positive net gain (excluding 0)
    fill_candidates = [c for c in range(1, 10) if diff[c] > 0]
    if len(fill_candidates) != 1:
        return None
    fill_color = fill_candidates[0]
    # frame_color: the most common non-zero color in the input
    in_palette = [(c, in_counts[c]) for c in range(1, 10) if in_counts[c] > 0]
    if not in_palette:
        return None
    in_palette.sort(key=lambda t: -t[1])
    frame_color = in_palette[0][0]
    src = (f"def solve(g):\n"
           f"    return fill_enclosed(g, {frame_color}, {fill_color})\n")
    if verify_program(src, task):
        return src
    return None


def _kron_with_k_source(task) -> str | None:
    """Infer the integer k for kron_tile / masked_kron_tile / brickwall_tile
    from a single train pair, and try each variant until one verifies.
    """
    from .verifier import verify_program
    pair = task.train[0]
    inp = np.array(pair["input"], dtype=int)
    out = np.array(pair["output"], dtype=int)
    if inp.shape == out.shape:
        return None
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    kh, kw = oh // ih, ow // iw
    if kh != kw:
        # rectangular tiling is rarer; we can still try with the row scale
        # (LLM-style) but it's not a primitive we have. Skip.
        return None
    k = kh
    for name in ("kron_tile", "masked_kron_tile", "brickwall_tile"):
        src = f"def solve(g):\n    return {name}(g, {k})\n"
        if verify_program(src, task):
            return src
    return None


def _single_color_recolor_source(task) -> str | None:
    """If the output has exactly one non-zero color across the whole grid,
    and we can express the transformation as a colormap on the input
    (possibly composed with an orientation transform), emit the solve().

    The 'object recolor by marker' family: input has a 'shape' color and a
    'marker' color; output has the shape recolored to the marker's color
    and the marker removed (set to 0). This pattern shows up ~60 times in
    ARC-AGI-2 and pure-DSL would otherwise miss it.
    """
    from .verifier import verify_program
    pair = task.train[0]
    inp = np.array(pair["input"], dtype=int)
    out = np.array(pair["output"], dtype=int)
    if inp.shape != out.shape:
        return None
    out_palette = set(np.unique(out).tolist()) - {0}
    if len(out_palette) != 1:
        return None
    target = next(iter(out_palette))
    # Try the standard 'object recolor' pattern: src=marker_color (input has
    # only one marker cell) maps to 0, and the dominant non-zero color maps
    # to target. Test against ALL train pairs (verify_program does that).
    in_palette = [c for c in range(1, 10) if (inp == c).any()]
    if len(in_palette) < 2:
        return None
    counts = {c: int((inp == c).sum()) for c in in_palette}
    # Shape color = most common, marker color = least common
    sorted_by_count = sorted(counts.items(), key=lambda t: t[1])
    marker = sorted_by_count[0][0]
    shape = sorted_by_count[-1][0]
    f = [0] * 10
    f[shape] = target
    src = (f"def solve(g):\n"
           f"    _f = np.array({f}, dtype=np.int64)\n"
           f"    return _f[g]\n")
    if verify_program(src, task):
        return src
    # Fallback: try every (source_color -> target, everything else -> 0)
    for sc in in_palette:
        if sc == target:
            continue
        f = [0] * 10
        f[sc] = target
        src = (f"def solve(g):\n"
               f"    _f = np.array({f}, dtype=np.int64)\n"
               f"    return _f[g]\n")
        if verify_program(src, task):
            return src
    # Try recolor + orientation: for each orientation, apply it first, then
    # the same colormap. Catches 'recolor AND shift/rotate' cases.
    orientations = [
        ("", "g"),
        ("rotate_cw", "rotate_cw(g)"),
        ("rotate_ccw", "rotate_ccw(g)"),
        ("rotate_cw(rotate_cw", "rotate_cw(rotate_cw(g))"),
        ("flip_h", "flip_h(g)"),
        ("flip_v", "flip_v(g)"),
    ]
    for _, expr in orientations:
        f = [0] * 10
        # Build a map: every non-zero input color -> target
        for c in in_palette:
            f[c] = target
        src = (f"def solve(g):\n"
               f"    _f = np.array({f}, dtype=np.int64)\n"
               f"    return _f[{expr}]\n")
        if verify_program(src, task):
            return src
    return None


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


def synthesize(task, max_compose: bool = True, verbose: bool = False,
               use_cache: bool = True) -> str | None:
    """Bounded program synthesizer over the primitive library.

    Tries (in order, cheapest first): cache -> single primitives -> colormap ->
    specialized probes (fill_enclosed, kron_with_k) -> 2-primitive compositions.
    Returns the first source that verifies against all train pairs, or None.
    This is the symbolic floor; the LLM extends beyond it.

    If `use_cache` is True (default), the task's structural fingerprint is
    checked against an on-disk cache first; on a hit we re-verify (in case
    primitives changed) and short-circuit. On a miss we run the full search
    and write the verified source to the cache.
    """
    from .verifier import verify_program
    from .cache import fingerprint as _fp, get_cached as _get, put_cached as _put
    # 0. Cache lookup (re-verify, then return if it still works)
    if use_cache:
        cached = _get(task)
        if cached and verify_program(cached, task):
            if verbose:
                print("  cache hit")
            return cached
    # 1. Single-primitive programs (cheapest)
    for src in _single_sources():
        if verify_program(src, task):
            if verbose:
                print("  verified (single):", src.strip().splitlines()[0])
            if use_cache:
                _put(task, src)
            return src
    # 2. Color-map probe (cheap, only same-shape tasks)
    cm = _color_map_source(task)
    if cm and verify_program(cm, task):
        if verbose:
            print("  verified (colormap)")
        if use_cache:
            _put(task, cm)
        return cm
    # 2b. Single-color-output recolor probe (object recolor by marker)
    sc = _single_color_recolor_source(task)
    if sc:
        if verbose:
            print("  verified (single_color_recolor)")
        if use_cache:
            _put(task, sc)
        return sc
    # 3. Specialized probes (frame+fill, kron-with-inferred-k)
    fe = _fill_enclosed_source(task)
    if fe:
        if verbose:
            print("  verified (fill_enclosed)")
        if use_cache:
            _put(task, fe)
        return fe
    kr = _kron_with_k_source(task)
    if kr:
        if verbose:
            print("  verified (kron_inferred_k)")
        if use_cache:
            _put(task, kr)
        return kr
    so = _shift_object_source(task)
    if so:
        if verbose:
            print("  verified (shift_object)")
        if use_cache:
            _put(task, so)
        return so
    # 4. 2-primitive compositions (more expensive)
    if max_compose:
        for src in _composition_sources():
            if verify_program(src, task):
                if verbose:
                    print("  verified (compose):", src.strip().splitlines()[0])
                if use_cache:
                    _put(task, src)
                return src
    return None


def search_solve(task, verbose: bool = False, use_cache: bool = True) -> str | None:
    """Backward-compatible alias for synthesize()."""
    return synthesize(task, verbose=verbose, use_cache=use_cache)


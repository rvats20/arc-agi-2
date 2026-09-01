"""Failure-mode analysis: categorize the 986 unsolved training tasks.

For each task, compute cheap structural features that hint at the dominant
transformation family. This tells us which new primitives to add for the
biggest score jump.

Output: analysis/failure_modes.json + a printed summary.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arc_agi2 import search_solve, verify_program
from arc_agi2.loader import load_all


def shape_features(inp: np.ndarray, out: np.ndarray) -> dict:
    feats = {}
    feats["shape_same"] = inp.shape == out.shape
    feats["shape_factor"] = None
    if inp.shape == out.shape:
        feats["shape_factor"] = "same"
    else:
        oh, ow = inp.shape
        nh, nw = out.shape
        if nh % oh == 0 and nw % ow == 0 and nh // oh == nw // ow:
            feats["shape_factor"] = f"scale_{nh // oh}"
        elif oh % nh == 0 and ow % nw == 0 and oh // nh == ow // nw:
            feats["shape_factor"] = f"downscale_{oh // nh}"
        else:
            feats["shape_factor"] = f"free_{oh}x{ow}->{nh}x{nw}"
    return feats


def color_features(inp: np.ndarray, out: np.ndarray) -> dict:
    feats = {}
    # Same-shape: check consistent color remap (0..9 -> 0..9)
    if inp.shape == out.shape:
        flat_in = inp.ravel().tolist()
        flat_out = out.ravel().tolist()
        mapping = {}
        consistent = True
        for a, b in zip(flat_in, flat_out):
            if a in mapping and mapping[a] != b:
                consistent = False
                break
            mapping[a] = b
        feats["colormap_consistent"] = consistent
        feats["colormap_size"] = len(mapping)
    else:
        feats["colormap_consistent"] = None
        feats["colormap_size"] = None
    return feats


def symmetry_features(inp: np.ndarray, out: np.ndarray) -> dict:
    """Did the output mirror / rotate the input?"""
    feats = {}
    feats["out_eq_in"] = np.array_equal(inp, out)
    if inp.shape == out.shape:
        feats["out_eq_rot_cw"] = np.array_equal(out, np.rot90(inp, -1))
        feats["out_eq_rot_ccw"] = np.array_equal(out, np.rot90(inp, 1))
        feats["out_eq_flip_h"] = np.array_equal(out, np.fliplr(inp))
        feats["out_eq_flip_v"] = np.array_equal(out, np.flipud(inp))
        feats["out_eq_transpose"] = np.array_equal(out, inp.T)
    else:
        feats["out_eq_rot_cw"] = None
        feats["out_eq_rot_ccw"] = None
        feats["out_eq_flip_h"] = None
        feats["out_eq_flip_v"] = None
        feats["out_eq_transpose"] = None
    return feats


def content_features(inp: np.ndarray, out: np.ndarray) -> dict:
    """Density, palette, count of non-zero cells."""
    feats = {}
    feats["in_nz"] = int((inp != 0).sum())
    feats["out_nz"] = int((out != 0).sum())
    feats["in_palette"] = int(len(np.unique(inp)))
    feats["out_palette"] = int(len(np.unique(out)))
    # 'blank' input with a single color seed
    feats["in_blank"] = bool((inp == 0).all())
    feats["out_blank"] = bool((out == 0).all())
    return feats


def diagnose_task(task) -> dict:
    """Run cheap probes against all train pairs; aggregate verdict per probe."""
    # Use first pair as the representative for the per-task verdict
    pair = task.train[0]
    inp = np.array(pair["input"], dtype=int)
    out = np.array(pair["output"], dtype=int)
    feats = {
        "n_train": len(task.train),
        "shape": shape_features(inp, out),
        "color": color_features(inp, out),
        "symmetry": symmetry_features(inp, out),
        "content": content_features(inp, out),
    }
    # Check same-shape and consistent colormap across ALL train pairs
    all_same_shape = all(
        np.array_equal(np.array(p["input"]).shape, np.array(p["output"]).shape)
        for p in task.train
    )
    all_consistent = True
    if all_same_shape:
        mapping = {}
        for p in task.train:
            fi = np.array(p["input"]).ravel()
            fo = np.array(p["output"]).ravel()
            for a, b in zip(fi, fo):
                if a in mapping and mapping[a] != b:
                    all_consistent = False
                    break
                mapping[a] = b
            if not all_consistent:
                break
    else:
        all_consistent = False
    feats["all_same_shape"] = all_same_shape
    feats["all_consistent_colormap"] = all_consistent
    # Can DSL solve it?
    src = search_solve(task)
    feats["dsl_solved"] = bool(src and verify_program(src, task))
    return feats


def main() -> None:
    tasks = load_all(HERE / "data", split="training")
    print(f"Analyzing {len(tasks)} training tasks...")

    n_solved = 0
    by_factor = Counter()
    by_colormap = Counter()
    by_out_eq_in = Counter()
    by_dens = Counter()
    failures: list[dict] = []

    for tid, task in tasks.items():
        f = diagnose_task(task)
        if f["dsl_solved"]:
            n_solved += 1
            continue
        failures.append({"tid": tid, **f})
        by_factor[f["shape"]["shape_factor"]] += 1
        by_colormap["consistent" if f["color"]["colormap_consistent"] else "no" if f["color"]["colormap_consistent"] is False else "shape_change"] += 1
        by_out_eq_in["eq_in" if f["symmetry"]["out_eq_in"] else "diff"] += 1
        in_nz = f["content"]["in_nz"]
        if in_nz == 0:
            d = "blank_in"
        elif in_nz < 10:
            d = "tiny"
        elif in_nz < 50:
            d = "small"
        elif in_nz < 200:
            d = "medium"
        else:
            d = "large"
        by_dens[d] += 1

    print(f"\n=== Summary ===")
    print(f"DSL solved: {n_solved}/{len(tasks)} = {n_solved/len(tasks):.2%}")
    print(f"Failures: {len(failures)}")
    print(f"\nShape factors among failures:")
    for k, v in by_factor.most_common():
        print(f"  {k}: {v}")
    print(f"\nSame-shape colormap-consistent among failures:")
    for k, v in by_colormap.most_common():
        print(f"  {k}: {v}")
    print(f"\nOutput == Input? (among failures)")
    for k, v in by_out_eq_in.most_common():
        print(f"  {k}: {v}")
    print(f"\nInput density bucket (among failures):")
    for k, v in by_dens.most_common():
        print(f"  {k}: {v}")

    out_path = HERE / "analysis" / "failure_modes.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "n_total": len(tasks),
        "n_solved": n_solved,
        "n_failed": len(failures),
        "by_factor": dict(by_factor),
        "by_colormap": dict(by_colormap),
        "by_out_eq_in": dict(by_out_eq_in),
        "by_density": dict(by_dens),
        "failures": failures,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Augmented-view scoring (NVARC/ARChitects style).

Winners score each candidate across D4 geometric views + color permutations
and pick the consensus winner (kgmon / probmul). This module is CPU-safe:
it re-uses the verifier to check candidates on augmented train pairs.
"""
from __future__ import annotations

import itertools
import numpy as np


def d4_views(g: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "id": g,
        "rot90": np.rot90(g, 1),
        "rot180": np.rot90(g, 2),
        "rot270": np.rot90(g, 3),
        "flip_h": np.fliplr(g),
        "flip_v": np.flipud(g),
        "transpose": np.asarray(g).T.copy(),
        "anti": np.fliplr(np.rot90(g, 1)),
    }


def _apply_color_perm(g: np.ndarray, perm: list[int]) -> np.ndarray:
    p = np.asarray(perm, dtype=int)
    return p[np.asarray(g, dtype=int)]


def score_candidate_augmented(src: str, train_pairs: list[dict],
                              n_color_perms: int = 3) -> tuple[float, int]:
    """Return (aug_score, n_pass). Lower aug_score = better.

    Runs src on original + D4 views + a few color perms of each train pair.
    n_pass = number of augmented views reproduced exactly.
    aug_score = total cell-diff (shape mismatch = heavy penalty).
    """
    from .verifier import run_program

    rng = np.random.default_rng(0)
    total_diff = 0
    n_pass = 0
    n_views = 0
    for pair in train_pairs:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        views = list(d4_views(inp).items())
        for vname, vin in views:
            vout_expected = d4_views(out).get(vname)
            if vout_expected is None or vin.shape != vout_expected.shape:
                continue
            perms = [[*range(10)]]
            for _ in range(n_color_perms):
                perm = [*range(10)]
                # keep 0 fixed (background), permute 1..9
                tail = perm[1:]
                rng.shuffle(tail)
                perm[1:] = list(tail)
                perms.append(perm)
            for perm in perms:
                pin = _apply_color_perm(vin, perm)
                pexp = _apply_color_perm(vout_expected, perm)
                n_views += 1
                try:
                    pred = np.asarray(run_program(src, pin), dtype=int)
                except Exception:
                    total_diff += 10 ** 6
                    continue
                if pred.shape != pexp.shape:
                    total_diff += abs(pred.size - pexp.size) + 10 ** 3
                else:
                    d = int((pred != pexp).sum())
                    total_diff += d
                    if d == 0:
                        n_pass += 1
    score = total_diff / max(n_views, 1)
    return score, n_pass


def rank_candidates(cands: list[str], train_pairs: list[dict]) -> list[tuple[float, int, str]]:
    """Sort candidates by (aug_score asc, n_pass desc). Returns list of tuples."""
    scored = []
    for src in cands:
        try:
            s, p = score_candidate_augmented(src, train_pairs)
        except Exception:
            s, p = float("inf"), 0
        scored.append((s, -p, src))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [(s, -negp, src) for s, negp, src in scored]

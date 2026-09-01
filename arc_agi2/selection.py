"""Selection / consensus algorithms for ARC-AGI-2 beam outputs.

For each task, we collect N candidate solutions from N beam views. We
need to pick the best 2 to put in attempt_1 and attempt_2.

Strategies ported from the public NVARC reference notebook (LB 33.89):

  score_full_probmul_3 - sum(3 - beam_score) + mean(sum(3 - aug_score))
    Higher is better. Uses cumulative NLL.

  score_kgmon - len(guesses) - mean(mean(score_aug))
    Higher is better. "Inverse consensus": rewards solutions that
    appear in MANY beam views, penalizes high aug-NLL (model confidence).

For ARC-AGI-2 (LB ~24-34% SOTA), kgmon typically wins. We use it as
the default selector.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Any
from .grid_text import hashable


def score_sum(guesses: Dict[str, dict], getter) -> List[np.ndarray]:
    """Group guesses by solution content, score each group, return solutions
    sorted by descending score."""
    grouped: Dict[tuple, list] = {}
    for g in guesses.values():
        h = hashable(g["solution"])
        grouped.setdefault(h, []).append(g)
    scored = [(getter(group), h) for h, group in grouped.items()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [np.asarray(h) for _, h in scored]


def getter_full_probmul_3(guesses: List[dict], baseline: float = 3.0) -> float:
    """Score = sum of (baseline - beam_score) + mean of summed (baseline - aug_score)."""
    inf_score = sum(baseline - g["beam_score"] for g in guesses)
    aug_score = float(np.mean([sum(baseline - s for s in g["score_aug"]) for g in guesses]))
    return inf_score + aug_score


def score_full_probmul_3(guesses: Dict[str, dict]) -> List[np.ndarray]:
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses: List[dict]) -> float:
    """Score = number of guesses - mean(mean(aug_scores)). Higher = better.
    Inverse consensus: many views agreeing on the same solution = good;
    high model confusion (high aug NLL) = bad."""
    return len(guesses) - float(np.mean([np.mean(g["score_aug"]) for g in guesses]))


def score_kgmon(guesses: Dict[str, dict]) -> List[np.ndarray]:
    return score_sum(guesses, getter_kgmon)


def _valid_sample(sample: dict) -> bool:
    """A beam output is valid if it has a 2D int array within size limits and
    finite, positive scores."""
    try:
        sol = np.asarray(sample["solution"])
        if sol.ndim != 2 or sol.size == 0:
            return False
        if sol.shape[0] > 30 or sol.shape[1] > 30:
            return False
        if not np.issubdtype(sol.dtype, np.integer):
            return False
        if sol.min() < 0 or sol.max() > 9:
            return False
        if not np.isfinite(sample["beam_score"]):
            return False
        if not len(sample.get("score_aug", [])):
            return False
        if not np.all(np.isfinite(sample["score_aug"])):
            return False
        return True
    except Exception:
        return False


def select_two(candidates: List[np.ndarray], varc_pred: np.ndarray = None) -> List[np.ndarray]:
    """Pick two distinct attempts from a list of candidate grids. If varc_pred
    is given and distinct, prefer (candidates[0], varc_pred). Otherwise return
    (candidates[0], candidates[1]) or (candidates[0], candidates[0]) if only
    one unique candidate exists."""
    out = []
    if candidates:
        out.append(candidates[0])
    if len(candidates) > 1:
        out.append(candidates[1])
    elif candidates:
        out.append(candidates[0])
    else:
        out.append(np.zeros((1, 1), dtype=int))
    if varc_pred is not None and varc_pred.ndim == 2 and varc_pred.size > 0:
        v = np.asarray(varc_pred, dtype=int)
        if v.shape[0] <= 30 and v.shape[1] <= 30 and not np.array_equal(v, out[0]):
            out[1] = v
    return out[:2]

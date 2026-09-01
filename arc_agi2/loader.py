"""Data loading for ARC-AGI-2.

ARC-AGI-2 ships per the official spec (2026) with these files:
  arc-agi_training-challenges.json   + arc-agi_training-solutions.json
  arc-agi_evaluation-challenges.json + arc-agi_evaluation-solutions.json
  arc-agi_test-challenges.json       (outputs withheld; private at eval)
  sample_submission.json

The loader is path-agnostic: point it at the directory holding these files
(env ARC_DATA_DIR or the `data_dir` argument). Filenames are matched by
glob so minor naming differences are tolerated.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR_ENV = "ARC_DATA_DIR"


@dataclass
class Task:
    task_id: str
    train: list[dict] = field(default_factory=list)   # [{"input","output"}]
    test: list[dict] = field(default_factory=list)    # [{"input"}] (+ "output" if known)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def _find_file(data_dir: Path, patterns: list[str]) -> Path | None:
    # Be tolerant of hyphen/underscore separators in the official filenames,
    # e.g. arc-agi_evaluation-challenges.json vs arc-agi_evaluation_challenges.json.
    import re
    all_patterns = []
    for pat in patterns:
        all_patterns.append(pat)
        # variant with separators flipped between words
        variant = re.sub(r"[-_]", lambda m: "_" if m.group(0) == "-" else "-", pat)
        if variant != pat:
            all_patterns.append(variant)
    for pat in all_patterns:
        hits = sorted(data_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENV)
    if data_dir is None:
        raise FileNotFoundError(
            f"No data directory given and ${DATA_DIR_ENV} is not set. "
            "Pass data_dir=... or export ARC_DATA_DIR=/path/to/arc-agi-2/data"
        )
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory does not exist: {p}")
    return p


def load_challenges(data_dir=None, split: str = "evaluation") -> dict[str, Any]:
    """split in {training, evaluation, test}. Returns raw challenge dict."""
    d = _resolve_data_dir(data_dir)
    name = "test" if split == "test" else split
    f = _find_file(d, [f"*{name}*-challenges.json", f"*{split}*-challenges.json"])
    if f is None:
        raise FileNotFoundError(f"Could not find {split} challenges JSON in {d}")
    return json.loads(f.read_text())


def load_solutions(data_dir=None, split: str = "evaluation") -> dict[str, Any]:
    d = _resolve_data_dir(data_dir)
    f = _find_file(d, [f"*{split}*-solutions.json"])
    if f is None:
        raise FileNotFoundError(f"Could not find {split} solutions JSON in {d}")
    return json.loads(f.read_text())


def load_task(task_id: str, data_dir=None, split: str = "evaluation") -> Task:
    challenges = load_challenges(data_dir, split)
    if task_id not in challenges:
        raise KeyError(f"{task_id} not in {split} challenges")
    raw = challenges[task_id]
    return Task(task_id=task_id, train=raw.get("train", []), test=raw.get("test", []))


def load_all(data_dir=None, split: str = "evaluation") -> dict[str, Task]:
    """Load every task for a split. For training/evaluation the solutions are
    merged into the test pairs' 'output' field when present (so we can score
    locally). Test split (private) has no outputs -> left None."""
    challenges = load_challenges(data_dir, split)
    out: dict[str, Task] = {}
    sol = None
    if split in ("training", "evaluation"):
        try:
            sol = load_solutions(data_dir, split)
        except FileNotFoundError:
            sol = None
    for tid, raw in challenges.items():
        test_pairs = [dict(p) for p in raw.get("test", [])]
        if sol is not None and tid in sol:
            gold = sol[tid]
            for i, g in enumerate(gold):
                if i < len(test_pairs):
                    test_pairs[i]["output"] = g
        out[tid] = Task(task_id=tid, train=raw.get("train", []), test=test_pairs)
    return out

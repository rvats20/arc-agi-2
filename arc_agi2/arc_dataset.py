"""ArcDataset: a key->task dictionary with augmentation support.

Each key encodes a transformation pipeline like
  'task_0.rot90.transpose.permute4150723698.ex021'
where each dot-separated op after the task name is applied to BOTH the
input AND output of the task. Used to:
  * Generate D4 (rotations) + S10 (color permutations) augmentations
  * Generate "ex" keys that rearrange the train pair order (LLM prompt
    ordering robustness)
  * Track per-key transformations so beam outputs can be inverted back
    to the canonical task form

Ported from the public NVARC + VARC reference notebooks.
"""
from __future__ import annotations
import json
import numpy as np
from typing import Optional, List, Dict, Any
from .grid_text import permute_mod, random_perm_descriptor, is_valid_solution


def _shuffled(lst):
    return np.random.permutation(lst).tolist()


class ArcDataset:
    """A dictionary of {key: {train: [...], test: [...]}} with augmentation
    operators that derive new keys from existing ones."""

    @staticmethod
    def forward_mod(a, key: str, use_perm: bool = True):
        """Apply each op in the key (after the first dot) to a grid a."""
        if a is None:
            return a
        for op in key.split(".")[1:]:
            if op == "rot90":
                a = np.rot90(a)
            elif op == "transpose":
                a = np.swapaxes(a, 0, 1)
            elif op.startswith("permute"):
                a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith("copy") or op.startswith("out") or op.startswith("ex") or op.startswith("run") or op.startswith("base"):
                pass
            else:
                raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key: str, inv_perm: bool = True):
        """Invert each op in reverse order, mapping a back to the canonical grid."""
        if a is None:
            return a
        for op in key.split(".")[1:][::-1]:
            if op == "rot90":
                a = np.rot90(a, k=3)
            elif op == "transpose":
                a = np.swapaxes(a, 0, 1)
            elif op.startswith("permute"):
                a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith("copy") or op.startswith("out") or op.startswith("ex") or op.startswith("run") or op.startswith("base"):
                pass
            else:
                raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies=None, keys=None, is_orig=False):
        replies = replies or {}
        if keys is not None:
            keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, path: str, keys=None) -> "ArcDataset":
        with open(path) as f:
            data = json.load(f)
        return cls(queries=data, is_orig=True, keys=keys)

    def load_replies(self, path: str) -> "ArcDataset":
        with open(path) as f:
            data = json.load(f)
        self.replies = {k: data[k] for k in self.keys if k in data}
        return self

    def shuffled(self) -> "ArcDataset":
        return self.__class__(queries=self.queries, replies=self.replies,
                              keys=_shuffled(self.keys))

    def mod(self, mod_func, descriptor=None, n: int = 1, stack=None,
            keep: bool = False, keep_key: bool = False,
            shuffle: bool = False, join: bool = True, inputs_only: bool = False):
        """Apply mod_func to every task, creating n derived keys. The descriptor
        is appended to the key after the function name. Used for rotations,
        transpositions, and permutations. If join=False, returns a list of
        ArcDataset (one per mod iteration)."""
        assert not (keep and keep_key)
        cur = self
        ret: list = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None:
            stack = mod_func.__name__.startswith("rot")
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i,
                                                        keep_key=keep_key,
                                                        inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        if not join:
            return ret
        return self.__class__.append(*ret)

    def mod_single(self, mod_func, descriptor, i: int, keep_key: bool, inputs_only: bool) -> "ArcDataset":
        queries = {}
        replies = {}
        keys = []
        for k0 in self.keys:
            # Build the descriptor string for this iteration.
            # The original NVARC code has special logic: if mod_func is np.copy
            # OR if descriptor is None, use the function name; if descriptor
            # is a string, use it as the descriptor; if it's callable, call
            # it (with or without the task dict, depending on its arity).
            if mod_func is np.copy and descriptor is None:
                desc_str = "copy{i}".format(i=i)
            elif descriptor is None:
                desc_str = mod_func.__name__
            elif isinstance(descriptor, str):
                desc_str = descriptor
            elif callable(descriptor):
                # Try calling with the task dict first (legacy); fall back
                # to no args (NVARC convention: permute_rnd_all_ takes no args).
                import inspect
                try:
                    if inspect.signature(descriptor).parameters:
                        desc_str = str(descriptor(self.queries[k0]))
                    else:
                        desc_str = str(descriptor())
                except (ValueError, TypeError):
                    # No signature info — try no args first
                    try:
                        desc_str = str(descriptor())
                    except TypeError:
                        desc_str = str(descriptor(self.queries[k0]))
            else:
                desc_str = str(descriptor)
            if "{i}" in desc_str:
                desc_str = desc_str.format(i=i)
            def func(a, d=desc_str):
                if descriptor is None or isinstance(descriptor, str):
                    return np.asarray(mod_func(a)).tolist()
                return np.asarray(mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'' if not inputs_only else 'I'}{desc_str}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a) if t == "input" or not inputs_only else a)
                                 for t, a in x.items()}
                                for x in e]
                           for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a) for a in self.replies[k0]]
        return self.__class__(queries=queries, replies=replies, keys=keys)

    def shuffle_ex(self, perm: Optional[np.ndarray] = None, keep_max: Optional[int] = None) -> "ArcDataset":
        """Reorder the train pairs of each task. perm is a per-task permutation;
        if None, a random one is generated."""
        new_keys = []
        new_queries = {}
        new_replies = {}
        for k in self.keys:
            n = len(self.queries[k]["train"])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None:
                p = p[:keep_max]
            new_k = f"{k}.ex" + ("-" if (p.max() > 9) else "") + "".join(map(str, p.tolist()))
            new_keys.append(new_k)
            new_queries[new_k] = {m: (np.array(v, dtype=object)[p].tolist() if m == "train" else v)
                                  for m, v in self.queries[k].items()}
            if k in self.replies:
                new_replies[new_k] = self.replies[k]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n: int = 1, shfl_keys: bool = False, seed: int = 42) -> "ArcDataset":
        """Generate D4 (rotations + transpose) + n random color permutations
        + per-task train-pair shuffle, in that order."""
        np.random.seed(seed)
        d: ArcDataset = self
        d = d.mod(np.transpose, keep=True)  # type: ignore[assignment]
        d = d.mod(np.rot90, n=3, keep=True)  # type: ignore[assignment]
        d = d.mod(permute_mod, random_perm_descriptor, n=n, shuffle=shfl_keys, keep=False)  # type: ignore[assignment]
        d = d.shuffle_ex()  # type: ignore[assignment]
        return d

    def split_multi_replies(self) -> "ArcDataset":
        """For tasks with multiple test inputs, generate one key per test
        (e.g. 'task_0_0', 'task_0_1'). Needed for per-test beam evaluation."""
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]["test"]))]
        return self.__class__(
            keys=[f"{k}_{i}" for k, i in key_indices],
            queries={f"{k}_{i}": {"train": self.queries[k]["train"],
                                   "test": [self.queries[k]["test"][i]]}
                     for k, i in key_indices},
            replies={f"{k}_{i}": [self.replies[k][i]]
                     for k, i in key_indices if k in self.replies},
        )

    def get_submission(self, results=None) -> dict:
        """Build the official submission.json structure: {task_id: [{attempt_1, attempt_2}, ...]}."""
        assert self.is_orig, "Must be run on original dataset."
        submission = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]}
                       for _ in range(len(self.queries[k]["test"]))]
                      for k in self.keys}
        if results is not None:
            self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results: dict, submission: dict) -> None:
        for k, v in results.items():
            base_id, base_nr = k.split("_")
            target = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target)]):
                target[f"attempt_{i+1}"] = np.asarray(g, dtype=int).tolist()

    def validate_submission(self, submission: dict) -> float:
        """Fraction of tasks where EITHER attempt matches ground truth. (0..1]"""
        assert self.is_orig, "Must be run on original dataset."
        score = 0.0
        for k in self.replies:
            for i, r in enumerate(self.replies[k]):
                for attempt in ("attempt_1", "attempt_2"):
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / max(len(self.replies[k]), 1)
                        break
        return score

    @staticmethod
    def append(*datasets) -> "ArcDataset":
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys=[k for d in datasets for k in d.keys],
        )

"""Per-task test-time priming for the LLM proposer (ARChitects/NVARC style).

On GPU (Kaggle), briefly fine-tune the proposer on the task's own train
pairs (augmented D4 + color perms) before generating candidates. On CPU
this module imports fine but prime() raises — callers must guard.

This mirrors what VARC already does for the ViT (60 TTT steps), extended
to the Qwen branch. LoRA-only, few steps, tiny LR — seconds per task.
"""
from __future__ import annotations

from typing import Any


def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError("ttt needs torch (Kaggle GPU)") from e


def build_priming_texts(task, n_aug: int = 8) -> list[str]:
    """Flatten train pairs (+ leicht augments) into chat text for SFT priming."""
    import numpy as np
    from .models_nvarc import _grid_to_text, _build_messages
    texts = []
    for p in task.train:
        texts.append((_grid_to_text(p["input"]), _grid_to_text(p["output"])))
    # D4 augments of pair 0 to fill budget
    if task.train:
        from .augment import d4_views
        inp = np.asarray(task.train[0]["input"], dtype=int)
        out = np.asarray(task.train[0]["output"], dtype=int)
        for name, vin in list(d4_views(inp).items())[:n_aug]:
            vout = d4_views(out).get(name)
            if vout is not None and vin.shape == vout.shape:
                texts.append((_grid_to_text(vin.tolist()),
                              _grid_to_text(vout.tolist())))
    return [f"IN:\n{i}\nOUT:\n{o}" for i, o in texts]


def prime_proposer(proposer: Any, task, steps: int = 8, lr: float = 1e-5) -> None:
    """LoRA TTT on the task's own pairs. No-op marker on CPU (raises)."""
    require_torch()
    import torch
    model = getattr(proposer, "_model", None)
    tok = getattr(proposer, "_tokenizer", None)
    if model is None or tok is None:
        return
    try:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                         lora_dropout=0.05, task_type="CAUSAL_LM")
        try:
            model = get_peft_model(model, cfg)
        except Exception:
            pass  # already LoRA-adapted
    except ImportError:
        pass  # full-model micro-finetune fallback (few steps, tiny LR)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    texts = build_priming_texts(task)
    for _ in range(steps):
        for t in texts:
            ids = tok(t, return_tensors="pt").to(model.device)["input_ids"]
            out = model(input_ids=ids, labels=ids)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
    model.eval()
    proposer._model = model

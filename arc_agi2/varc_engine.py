"""VARC engine: a small Vision Transformer with per-task test-time training.

Ported from the public reference notebook (LB 33.89 lineage). The VARC
predictor is COMPLEMENTARY to the LLM-based predictor — it produces
one candidate per test input via 60 gradient steps on a tiny custom
ViT. The score_kgmon consensus then merges VARC + Qwen candidates.

The VARC engine is heavy on torch (ViT, attention, 60 TTT steps per
task). On CPU-only machines (WSL), `import varc_engine` succeeds but
constructing an ARCViT raises RuntimeError via require_torch().

Design (from the reference):
  - Canvas: 30x30 with pad_val=10 (background = 10 = unobserved)
  - Per-task: instantiate ARCViT(num_tasks=1) fresh
  - Train pairs are placed at 4 corner offsets (0,0)(2,2)(0,4)(4,0) so
    the ViT sees 4 variants per train pair; one offset per training step
  - 60 AdamW steps with grad-clip=1.0, lr=3e-4, weight_decay=1e-4
  - Inference: 4 corner offsets for the test input, take the MAJORITY
    of the 4 predicted grids (they often agree on hard tasks)
  - Returns the first matching-majority grid (or first if all differ)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any
import math

import numpy as np

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    _TORCH_OK = True
    torch = _torch  # type: ignore[assignment]
    nn = _nn  # type: ignore[assignment]
    F = _F  # type: ignore[assignment]
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def require_torch() -> None:
    if not _TORCH_OK:
        raise RuntimeError(
            "varc_engine needs torch; install with `pip install torch` "
            "(or run on the Kaggle kernel which has torch preinstalled)"
        )


# --- RoPE + ViT (matches the reference cell 4 of arc-prize-2026-solver) ---

if _TORCH_OK:


    def rotate_half(x: "torch.Tensor") -> "torch.Tensor":
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    class VisionRotaryEmbeddingFast(nn.Module):
        def __init__(self, dim: int, pt_seq_len: int = 16, theta: float = 10000.0, no_rope: int = 1) -> None:
            super().__init__()
            self.dim = dim
            self.no_rope = no_rope
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(pt_seq_len).float()
            freqs_h = torch.repeat_interleave(torch.outer(t, freqs), 2, dim=-1)
            freqs_w = torch.repeat_interleave(torch.outer(t, freqs), 2, dim=-1)
            H = W = pt_seq_len
            fh = freqs_h.unsqueeze(1).expand(H, W, -1)
            fw = freqs_w.unsqueeze(0).expand(H, W, -1)
            freqs_2d = torch.cat((fh, fw), dim=-1)
            self.register_buffer("freqs_cos", freqs_2d.reshape(H * W, -1).cos())
            self.register_buffer("freqs_sin", freqs_2d.reshape(H * W, -1).sin())

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            seq_len = t.shape[2]
            if self.no_rope > 0:
                prefix, patches = t[:, :, :self.no_rope, :], t[:, :, self.no_rope:, :]
                p_len = patches.shape[2]
                cos = self.freqs_cos[:p_len, :].unsqueeze(0).unsqueeze(0)
                sin = self.freqs_sin[:p_len, :].unsqueeze(0).unsqueeze(0)
                patches_rot = patches * cos + rotate_half(patches) * sin
                return torch.cat((prefix, patches_rot), dim=2)
            cos = self.freqs_cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
            sin = self.freqs_sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
            return t * cos + rotate_half(t) * sin


    class PatchEmbed(nn.Module):
        def __init__(self, img_size: int = 30, patch_size: int = 2, in_chans: int = 256, embed_dim: int = 256) -> None:
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.grid_size = img_size // patch_size
            self.num_patches = self.grid_size * self.grid_size
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x).flatten(2).transpose(1, 2)


    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                     max_seq_len: int = 226, dropout: float = 0.1, no_rope: int = 1) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.attn_dropout = nn.Dropout(dropout)
            self.proj_dropout = nn.Dropout(dropout)
            half_head_dim = self.head_dim // 2
            self.rotary = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=int(max_seq_len ** 0.5) + 4,
                no_rope=no_rope,
            )

        def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            b, s, _ = x.shape
            qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k = self.rotary(qkv[0]), self.rotary(qkv[1])
            v = qkv[2]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            attn = self.attn_dropout(torch.softmax(scores, dim=-1))
            ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, self.embed_dim)
            return self.proj_dropout(self.proj(ctx))


    class ARCTransformerEncoderLayer(nn.Module):
        def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_dim: int = 512,
                     dropout: float = 0.1, max_seq_len: int = 226, no_rope: int = 1) -> None:
            super().__init__()
            self.self_attn = MultiHeadSelfAttention(
                embed_dim=embed_dim, num_heads=num_heads, max_seq_len=max_seq_len,
                dropout=dropout, no_rope=no_rope,
            )
            self.dropout1, self.norm1 = nn.Dropout(dropout), nn.LayerNorm(embed_dim)
            self.linear1, self.activation = nn.Linear(embed_dim, mlp_dim), nn.GELU()
            self.dropout2, self.linear2 = nn.Dropout(dropout), nn.Linear(mlp_dim, embed_dim)
            self.dropout3, self.norm2 = nn.Dropout(dropout), nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = self.norm1(x + self.dropout1(self.self_attn(x, key_padding_mask=key_padding_mask)))
            x = self.norm2(x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x))))))
            return x


    class ARCViT(nn.Module):
        """The custom VARC ViT. Takes a 30x30 (canvas-sized) grid of int colors
        in [0..10] (10 = pad), predicts per-cell color."""

        def __init__(self, num_tasks: int = 1, image_size: int = 30, num_colors: int = 11,
                     embed_dim: int = 256, depth: int = 6, num_heads: int = 8,
                     mlp_dim: int = 512, dropout: float = 0.1, num_task_tokens: int = 1,
                     patch_size: int = 2) -> None:
            super().__init__()
            self.image_size, self.num_colors = image_size, num_colors
            self.embed_dim, self.patch_size = embed_dim, patch_size
            self.seq_length = (image_size // patch_size) ** 2
            self.num_task_tokens = num_task_tokens
            self.color_embed = nn.Embedding(num_colors, embed_dim)
            self.task_token_embed = nn.Embedding(num_tasks, embed_dim * num_task_tokens)
            self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim)
            total_seq_len = num_task_tokens + self.seq_length
            self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
            self.encoder = nn.ModuleList([
                ARCTransformerEncoderLayer(
                    embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                    dropout=dropout, max_seq_len=total_seq_len, no_rope=num_task_tokens,
                )
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_colors * (patch_size ** 2))
            nn.init.trunc_normal_(self.positional_embed, std=0.02)
            nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

        def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            b = pixel_values.size(0)
            tokens = self.color_embed(pixel_values.long())
            tokens = self.patch_embed(tokens.permute(0, 3, 1, 2)) + self.positional_embed[:, : self.seq_length, :]
            task_tokens = self.task_token_embed(task_ids.long()).reshape(b, self.num_task_tokens, -1)
            h = torch.cat([task_tokens, tokens], dim=1)
            key_padding_mask = None
            if attention_mask is not None:
                mp = attention_mask.reshape(b, self.image_size // self.patch_size,
                                            self.patch_size,
                                            self.image_size // self.patch_size,
                                            self.patch_size)
                mp = mp.amax(dim=(2, 4)).reshape(b, self.seq_length)
                key_padding_mask = torch.cat(
                    [torch.zeros(b, self.num_task_tokens, device=pixel_values.device, dtype=torch.bool),
                     ~mp.bool()],
                    dim=1,
                )
            for layer in self.encoder:
                h = layer(h, key_padding_mask=key_padding_mask)
            pixel_states = self.norm(h)[:, self.num_task_tokens:, :]
            logits = self.head(pixel_states)
            G, P = self.image_size // self.patch_size, self.patch_size
            logits = logits.reshape(b, G, G, P, P, self.num_colors).permute(0, 1, 3, 2, 4, 5).reshape(
                b, self.image_size, self.image_size, self.num_colors,
            )
            return logits.permute(0, 3, 1, 2)


    class VARCCanvasProcessor:
        """Embeds a small ARC grid into a fixed 30x30 canvas with padding."""

        def __init__(self, canvas_size: int = 30, pad_val: int = 10) -> None:
            self.canvas_size, self.pad_val = canvas_size, pad_val

        def grid_to_canvas(self, grid: np.ndarray, offset_r: int = 0, offset_c: int = 0):
            H, W = grid.shape
            canvas = np.full((self.canvas_size, self.canvas_size), self.pad_val, dtype=np.int64)
            mask = np.zeros((self.canvas_size, self.canvas_size), dtype=bool)
            r_start = max(0, min(offset_r, self.canvas_size - H))
            c_start = max(0, min(offset_c, self.canvas_size - W))
            canvas[r_start:r_start + H, c_start:c_start + W] = grid
            mask[r_start:r_start + H, c_start:c_start + W] = True
            return canvas, mask

        def canvas_to_grid(self, canvas: np.ndarray, target_h: int, target_w: int,
                           offset_r: int = 0, offset_c: int = 0) -> np.ndarray:
            r_start = max(0, min(offset_r, self.canvas_size - target_h))
            c_start = max(0, min(offset_c, self.canvas_size - target_w))
            extracted = canvas[r_start:r_start + target_h, c_start:c_start + target_w]
            return np.where(extracted == self.pad_val, 0, extracted).astype(int)


    class VARCSolver:
        """Wraps an ARCViT. solve_task() runs 60 TTT steps + majority vote."""

        def __init__(self, model_weights_path: Optional[str] = None, device: str = "cpu",
                     canvas_size: int = 30, ttt_steps: int = 60, lr: float = 3e-4) -> None:
            self.device = torch.device(device)
            self.canvas_size, self.ttt_steps, self.lr = canvas_size, ttt_steps, lr
            self.processor = VARCCanvasProcessor(canvas_size=canvas_size)
            self.model = ARCViT(num_tasks=1, image_size=canvas_size, num_colors=11).to(self.device)
            if model_weights_path:
                try:
                    state = torch.load(model_weights_path, map_location=self.device)
                    self.model.load_state_dict(state)
                    print(f"[VARC] loaded weights from {model_weights_path}")
                except Exception as e:
                    print(f"[VARC] could not load weights from {model_weights_path}: {e}; using random init")
            self.model.eval()

        def _train_one(self, train_pairs: List[Dict[str, np.ndarray]]) -> "ARCViT":
            """Instantiate a fresh task-specific ViT and TTT for this task."""
            task_model = ARCViT(num_tasks=1, image_size=self.canvas_size, num_colors=11).to(self.device)
            task_model.load_state_dict(self.model.state_dict())
            task_model.train()
            opt = torch.optim.AdamW(task_model.parameters(), lr=self.lr, weight_decay=1e-4)
            inps, tgts, masks = [], [], []
            for pair in train_pairs:
                inp_g = np.asarray(pair["input"])
                tgt_g = np.asarray(pair["output"])
                for (dr, dc) in [(0, 0), (2, 2), (0, 4), (4, 0)]:
                    c_inp, m_inp = self.processor.grid_to_canvas(inp_g, offset_r=dr, offset_c=dc)
                    c_tgt, _ = self.processor.grid_to_canvas(tgt_g, offset_r=dr, offset_c=dc)
                    inps.append(torch.tensor(c_inp, dtype=torch.long))
                    tgts.append(torch.tensor(c_tgt, dtype=torch.long))
                    masks.append(torch.tensor(m_inp, dtype=torch.bool))
            if not inps:
                return task_model
            bi = torch.stack(inps).to(self.device)
            bt = torch.stack(tgts).to(self.device)
            bm = torch.stack(masks).to(self.device)
            task_ids = torch.zeros(len(inps), dtype=torch.long, device=self.device)
            for _ in range(self.ttt_steps):
                opt.zero_grad(set_to_none=True)
                logits = task_model(bi, task_ids, attention_mask=bm)
                loss = F.cross_entropy(logits, bt, ignore_index=10)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1.0)
                opt.step()
            return task_model

        def solve_task(self, train_pairs: List[Dict[str, np.ndarray]], test_input: np.ndarray,
                       target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
            """Predict the output grid for a single test input. Returns a numpy int array."""
            if not target_shape:
                out_shapes = [np.asarray(p["output"]).shape for p in train_pairs]
                target_shape = out_shapes[0] if len(set(out_shapes)) == 1 else test_input.shape
            task_model = self._train_one(train_pairs)
            task_model.eval()
            preds = []
            with torch.no_grad():
                for (dr, dc) in [(0, 0), (1, 1), (2, 0), (0, 2)]:
                    t_canvas, t_mask = self.processor.grid_to_canvas(test_input, offset_r=dr, offset_c=dc)
                    t_inp = torch.tensor(t_canvas, dtype=torch.long, device=self.device).unsqueeze(0)
                    t_m = torch.tensor(t_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                    out_logits = task_model(t_inp,
                                             torch.zeros(1, dtype=torch.long, device=self.device),
                                             attention_mask=t_m)
                    pred_canvas = out_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                    pred_grid = self.processor.canvas_to_grid(pred_canvas, target_shape[0], target_shape[1],
                                                             offset_r=dr, offset_c=dc)
                    preds.append(pred_grid)
            # Majority vote across the 4 offset predictions
            hashes = [tuple(map(tuple, p)) for p in preds]
            best = max(set(hashes), key=hashes.count)
            for p in preds:
                if tuple(map(tuple, p)) == best:
                    return p
            return preds[0]

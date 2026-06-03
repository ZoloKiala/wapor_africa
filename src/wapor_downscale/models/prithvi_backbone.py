"""Real Prithvi-EO-2.0-300M backbone loader.

Imports the official model code from third_party/prithvi_eo_v2_300m/prithvi_mae.py
(downloaded from huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M).

Loading path:
    weights file = third_party/prithvi_eo_v2_300m/Prithvi_EO_V2_300M.pt  (1.3 GB)
    config file  = third_party/prithvi_eo_v2_300m/config.json

A MockViTStem fallback is kept for environments without the checkpoint
(useful for CI / quick scaffolding tests).

Output shape contract (real and mock):
    input  : (B, 6, H, W)
    output : (B, embed_dim, H/16, W/16)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
PRITHVI_DIR = REPO_ROOT / "third_party" / "prithvi_eo_v2_300m"

# Make the official model code importable as a top-level module.
if PRITHVI_DIR.exists() and str(PRITHVI_DIR) not in sys.path:
    sys.path.insert(0, str(PRITHVI_DIR))


class MockViTStem(nn.Module):
    """Stand-in backbone with matching I/O shape. Useful for plumbing tests."""

    def __init__(self, in_chans: int = 6, embed_dim: int = 1024, patch_size: int = 16):
        super().__init__()
        assert patch_size == 16, "MockViTStem hard-coded for patch=16"
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        c1, c2, c3 = embed_dim // 8, embed_dim // 4, embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c1, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c3, embed_dim, 2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PrithviBackbone(nn.Module):
    """Wraps Prithvi-EO-2.0-300M's ViT encoder for downstream regression use.

    Args:
        img_size : matches the patch size you intend to feed (default 256).
        freeze_until : "none" trains everything; "all" freezes everything;
                       integer N (e.g. 22) freezes blocks [0..N-1], trains [N..depth-1] + final norm.
        mock : if True, skip Prithvi loading and use MockViTStem.
        strict_load : pass-through to load_state_dict; default False because
                      the published state_dict contains pos_embed buffers that we
                      drop (regenerated via sin-cos at init).
    """

    def __init__(
        self,
        img_size: int = 256,
        freeze_until: str | int = 22,   # blocks 22-23 + norm trainable by default
        mock: bool = False,
        strict_load: bool = False,
        ckpt_path: Path | None = None,
    ):
        super().__init__()
        self.using_mock = mock
        self.freeze_until = freeze_until
        if mock:
            self._model = MockViTStem(embed_dim=1024, patch_size=16)
            self.embed_dim = 1024
            self.patch_size = 16
            self.depth = 0
        else:
            self._build_real(img_size, ckpt_path, strict_load)
        self._apply_freeze()

    def _build_real(self, img_size: int, ckpt_path: Path | None, strict_load: bool) -> None:
        if not PRITHVI_DIR.exists():
            raise FileNotFoundError(
                f"Prithvi model code not found at {PRITHVI_DIR}. "
                "Download from huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M."
            )
        from prithvi_mae import PrithviViT  # type: ignore

        cfg = json.loads((PRITHVI_DIR / "config.json").read_text())
        p = cfg["pretrained_cfg"]
        self.embed_dim = p["embed_dim"]
        self.patch_size = p["patch_size"][1]
        self.depth = p["depth"]

        self._model = PrithviViT(
            img_size=img_size,
            num_frames=1,                      # single dekad input (no time dim)
            patch_size=tuple(p["patch_size"]),
            in_chans=p["in_chans"],
            embed_dim=p["embed_dim"],
            depth=p["depth"],
            num_heads=p["num_heads"],
            mlp_ratio=p["mlp_ratio"],
            coords_encoding=p.get("coords_encoding", []),
            coords_scale_learn=p.get("coords_scale_learn", False),
        )

        ckpt = Path(ckpt_path) if ckpt_path else (PRITHVI_DIR / "Prithvi_EO_V2_300M.pt")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        # State dict belongs to PrithviMAE (encoder + decoder). Strip encoder. prefix,
        # drop fixed pos_embed buffers (re-initialized via sin-cos at construction time),
        # discard everything decoder-related.
        encoder_sd = {}
        for k, v in sd.items():
            if "pos_embed" in k:
                continue
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder."):]] = v
        missing, unexpected = self._model.load_state_dict(encoder_sd, strict=strict_load)
        if missing or unexpected:
            print(f"[PrithviBackbone] load report: {len(missing)} missing, {len(unexpected)} unexpected")
            if unexpected:
                print(f"  unexpected keys (first 5): {unexpected[:5]}")
            if missing:
                print(f"  missing keys (first 5): {missing[:5]}")

    def _apply_freeze(self) -> None:
        if self.freeze_until == "none":
            return
        if self.freeze_until == "all" or (self.using_mock and self.freeze_until != "none"):
            for p in self._model.parameters():
                p.requires_grad_(False)
            return
        # Integer N: freeze patch_embed, cls_token, blocks [0..N-1].
        n = int(self.freeze_until)
        self._model.cls_token.requires_grad_(False)
        for p in self._model.patch_embed.parameters():
            p.requires_grad_(False)
        for i, blk in enumerate(self._model.blocks):
            req = i >= n
            for p in blk.parameters():
                p.requires_grad_(req)
        # final norm stays trainable

    def unfrozen_params(self) -> list[nn.Parameter]:
        return [p for p in self._model.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_mock:
            return self._model(x)
        # Real Prithvi: get the final-layer hidden state, drop CLS, reshape (B, L, C) -> (B, C, H, W).
        feats = self._model.forward_features(x)   # list of (B, 1+L, C), len=depth
        last = feats[-1][:, 1:, :]                # drop CLS -> (B, L, C)
        B, L, C = last.shape
        # L = T*H*W with T=1 since patch_size[0]=1 and num_frames=1, so L = H_p * W_p.
        side = int(L ** 0.5)
        return last.transpose(1, 2).reshape(B, C, side, side)

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return all 24 block outputs as (B, embed_dim, H/16, W/16) tensors.

        For V2 multi-block fusion. For the mock, returns 24 copies of the same
        stem output so downstream consumers can be tested without the real
        checkpoint.
        """
        if self.using_mock:
            last = self._model(x)
            return [last] * 24
        feats = self._model.forward_features(x)
        out = []
        for f in feats:
            f = f[:, 1:, :]
            B, L, C = f.shape
            side = int(L ** 0.5)
            out.append(f.transpose(1, 2).reshape(B, C, side, side))
        return out

"""End-to-end Prithvi-based regression model for WaPOR L3 AETI downscaling.

Architecture:
    9-channel input
        ├── [B4, B8, B11]                  -> BandAdapter (S2 -> HLS norm) -> Prithvi backbone -> (B, 1024, 16, 16)
        │                                                                       |
        |                                                              ConvTranspose 16x ->
        |                                                                  (B, 256, 256, 256)
        |                                                                       |
        └── [ETa300m, NDVI, NDMI, FVC, sinDOY, cosDOY] -> SideEncoder (3-conv CNN) -> (B, 64, 256, 256)
                                                                                       |
                                                  concat ->  (B, 320, 256, 256) -> Head -> (B, 1, 256, 256)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from wapor_downscale.models.band_adapter import S2ToHLSAdapter, select_side, select_spectral
from wapor_downscale.models.prithvi_backbone import PrithviBackbone


class SideEncoder(nn.Module):
    """Small CNN for non-spectral predictors. Keeps native resolution."""

    def __init__(self, in_chans: int = 6, out_chans: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, out_chans, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_chans, out_chans, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrithviRegression(nn.Module):
    """Full Prithvi-adapted regression model for AETI."""

    def __init__(
        self,
        img_size: int = 256,
        freeze_until: str | int = 22,    # int N freezes blocks 0..N-1; "all"/"none" supported
        side_chans: int = 6,
        side_dim: int = 64,
        out_chans: int = 1,
        mock_backbone: bool = False,
    ):
        super().__init__()
        self.band_adapter = S2ToHLSAdapter()
        self.backbone = PrithviBackbone(
            img_size=img_size, freeze_until=freeze_until, mock=mock_backbone,
        )
        self.side = SideEncoder(in_chans=side_chans, out_chans=side_dim)

        embed = self.backbone.embed_dim
        # Upsample backbone tokens (H/16, W/16) back to native resolution.
        # Two-stage ConvTranspose keeps params reasonable vs a single x16 transpose.
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed, embed // 2, 4, stride=4),   # x4
            nn.GELU(),
            nn.ConvTranspose2d(embed // 2, 256, 4, stride=4),     # x4 more -> x16 total
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256 + side_dim, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, out_chans, 1),
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        spec = select_spectral(stack)        # (B, 3, H, W)
        side = select_side(stack)            # (B, 6, H, W)
        hls = self.band_adapter(spec)        # (B, 6, H, W) normalized
        feat = self.backbone(hls)            # (B, embed, H/16, W/16)
        feat = self.upsample(feat)           # (B, 256, H, W)
        side_feat = self.side(side)          # (B, side_dim, H, W)
        out = self.head(torch.cat([feat, side_feat], dim=1))
        return out


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------- V2 -----------------------------
# Three improvements over V1, based on the Prithvi loss analysis:
#   1. Multi-block fusion: combine features from blocks {6, 12, 18, 24} instead of
#      only block 24 — early blocks carry low-level texture, late blocks carry
#      semantics. (UperNet pattern.)
#   2. PixelShuffle progressive upsampling 16 -> 32 -> 64 -> 128 -> 256, halving
#      channels at each step. Smoother than 16x ConvTranspose.
#   3. Input skip path: small CNN on the raw 9-channel stack joins the head, so
#      high-frequency signal (especially ETa300m) bypasses the ViT bottleneck.


class _PixelShuffleUp(nn.Module):
    """One stage of x2 PixelShuffle upsampling that halves channels."""

    def __init__(self, in_ch: int):
        super().__init__()
        out_ch = in_ch // 2
        self.expand = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.expand(x)))


class PrithviRegressionV2(nn.Module):
    """V2: multi-block fusion + PixelShuffle decoder + input skip.

    Fuses features from blocks {6, 12, 18, 24} via 1x1 laterals (UperNet),
    upsamples 16x via four PixelShuffle(2) stages, and concatenates a CNN over
    the raw input at native resolution before the head.
    """

    def __init__(
        self,
        img_size: int = 256,
        freeze_until: str | int = 22,
        fusion_blocks: tuple[int, ...] = (5, 11, 17, 23),  # 0-indexed: blocks 6, 12, 18, 24
        fusion_dim: int = 256,
        side_chans: int = 6,
        side_dim: int = 32,
        input_skip_chans: int = 9,
        input_skip_dim: int = 16,
        out_chans: int = 1,
        mock_backbone: bool = False,
    ):
        super().__init__()
        self.band_adapter = S2ToHLSAdapter()
        self.backbone = PrithviBackbone(
            img_size=img_size, freeze_until=freeze_until, mock=mock_backbone,
        )
        self.fusion_blocks = fusion_blocks
        embed = self.backbone.embed_dim  # 1024

        # Lateral 1x1 convs, one per fused block.
        self.lateral = nn.ModuleList([
            nn.Conv2d(embed, fusion_dim, 1) for _ in fusion_blocks
        ])
        # Small post-fusion conv after summing.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.GELU(),
        )
        # Progressive PixelShuffle x16: 16 -> 32 -> 64 -> 128 -> 256.
        # Channels: 256 -> 128 -> 64 -> 32 -> 16.
        c = fusion_dim
        ups = []
        for _ in range(4):
            stage = _PixelShuffleUp(c)
            ups.append(stage)
            c = stage.out_channels
        self.upsample = nn.ModuleList(ups)
        backbone_decoded_dim = c  # 16 by default

        # Non-spectral side encoder (same channel set as V1).
        self.side = nn.Sequential(
            nn.Conv2d(side_chans, side_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(side_dim, side_dim, 3, padding=1),
        )

        # Raw-input skip path: small CNN over the full 9-channel stack.
        self.input_skip = nn.Sequential(
            nn.Conv2d(input_skip_chans, input_skip_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(input_skip_dim, input_skip_dim, 3, padding=1),
        )

        head_in = backbone_decoded_dim + side_dim + input_skip_dim
        self.head = nn.Sequential(
            nn.Conv2d(head_in, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, out_chans, 1),
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        spec = select_spectral(stack)            # (B, 3, H, W)
        side_in = select_side(stack)             # (B, 6, H, W)
        hls = self.band_adapter(spec)            # (B, 6, H, W)

        all_feats = self.backbone.forward_features(hls)  # list[(B, 1024, H/16, W/16)]
        fused = sum(lat(all_feats[i]) for i, lat in zip(self.fusion_blocks, self.lateral))
        x = self.fusion_conv(fused)

        for stage in self.upsample:
            x = stage(x)                          # 16 -> ... -> 256

        side_feat = self.side(side_in)            # (B, side_dim, H, W)
        skip_feat = self.input_skip(stack)        # (B, input_skip_dim, H, W)

        return self.head(torch.cat([x, side_feat, skip_feat], dim=1))


# ----------------------------- V3 -----------------------------
# Targeted fix to V2's regressions, based on the V2 hold-out outcome (3.560 > V1's 3.519):
#   - Concat the 4 block laterals (not sum) so block-specific signal is preserved.
#   - Less aggressive PixelShuffle channel schedule (256 -> 192 -> 160 -> 128 -> 96)
#     so the head sees ~96 backbone channels instead of V2's 16.
#   - More backbone capacity unfrozen by default (last 6 blocks, freeze_until=18).
# Expected target: ~3.30 RMSE on hold-out (match SwinIR), not beat.


class _PixelShuffleUpExplicit(nn.Module):
    """PixelShuffle x2 stage with an explicit out_ch (not just halving)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.expand = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.expand(x)))


class PrithviRegressionV3(nn.Module):
    """V3: concat-fusion + thicker decoder + freeze=18 (default)."""

    def __init__(
        self,
        img_size: int = 256,
        freeze_until: str | int = 18,
        fusion_blocks: tuple[int, ...] = (5, 11, 17, 23),
        lateral_dim: int = 128,
        fusion_dim: int = 256,
        decoder_ch_schedule: tuple[int, ...] = (192, 160, 128, 96),  # after each PixelShuffle stage
        side_chans: int = 6,
        side_dim: int = 32,
        input_skip_chans: int = 9,
        input_skip_dim: int = 16,
        out_chans: int = 1,
        mock_backbone: bool = False,
    ):
        super().__init__()
        assert len(decoder_ch_schedule) == 4, "decoder needs 4 stages for 16x upsampling"
        self.band_adapter = S2ToHLSAdapter()
        self.backbone = PrithviBackbone(
            img_size=img_size, freeze_until=freeze_until, mock=mock_backbone,
        )
        self.fusion_blocks = fusion_blocks
        embed = self.backbone.embed_dim

        # 1x1 laterals, then CONCAT (not sum) -> compress to fusion_dim.
        self.lateral = nn.ModuleList([
            nn.Conv2d(embed, lateral_dim, 1) for _ in fusion_blocks
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(lateral_dim * len(fusion_blocks), fusion_dim, 1),
            nn.GELU(),
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.GELU(),
        )

        # Progressive PixelShuffle x16 with explicit per-stage channel schedule.
        ups = []
        c_in = fusion_dim
        for c_out in decoder_ch_schedule:
            ups.append(_PixelShuffleUpExplicit(c_in, c_out))
            c_in = c_out
        self.upsample = nn.ModuleList(ups)
        backbone_decoded_dim = c_in  # 96 by default

        self.side = nn.Sequential(
            nn.Conv2d(side_chans, side_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(side_dim, side_dim, 3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(input_skip_chans, input_skip_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(input_skip_dim, input_skip_dim, 3, padding=1),
        )

        head_in = backbone_decoded_dim + side_dim + input_skip_dim   # default: 96 + 32 + 16 = 144
        self.head = nn.Sequential(
            nn.Conv2d(head_in, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, out_chans, 1),
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        spec = select_spectral(stack)
        side_in = select_side(stack)
        hls = self.band_adapter(spec)

        all_feats = self.backbone.forward_features(hls)
        # CONCAT laterals (not sum).
        laterals = [lat(all_feats[i]) for i, lat in zip(self.fusion_blocks, self.lateral)]
        fused = torch.cat(laterals, dim=1)        # (B, lateral_dim * n_blocks, 16, 16)
        x = self.fusion_conv(fused)               # (B, fusion_dim, 16, 16)

        for stage in self.upsample:
            x = stage(x)                          # 16 -> 32 -> 64 -> 128 -> 256

        side_feat = self.side(side_in)
        skip_feat = self.input_skip(stack)
        return self.head(torch.cat([x, side_feat, skip_feat], dim=1))

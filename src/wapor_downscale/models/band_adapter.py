"""Map our 9-channel input stack to Prithvi-EO-2.0's expected 6-band HLS layout.

Our channel order (from unet_common.py):
    0:B4 (Red)  1:B8 (NIR)  2:B11 (SWIR1)  3:ETa300m  4:NDVI  5:NDMI  6:FVC  7:sinDOY  8:cosDOY

Prithvi-EO-2.0 v2 expects HLS bands in this order:
    Blue, Green, Red, NIR_Narrow, SWIR_1, SWIR_2

We only have B4/B8/B11. Two strategies:

(a) zero_fill (default, used here): Blue/Green/SWIR2 set to 0 AFTER normalization.
    This preserves the pretrained patch_embed weights at the cost of feeding the
    model values it never saw at pretraining. Cheapest to implement; reasonable
    starting point.

(b) learned adapter: a 1x1 conv from 3 -> 6 with trainable weights. Drops the
    zero assumption but breaks the bijection patch_embed expects. Worth trying
    later as an ablation.

Per-band normalization stats below are PLACEHOLDERS. Replace with the values
from the Prithvi-EO-2.0 config (`bands_norm` in the model card) once the
checkpoint is downloaded.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Real values from third_party/prithvi_eo_v2_300m/config.json, divided by 10000
# because our unet_common.stack_to_tensors already scales DN -> 0-1 reflectance,
# while the published Prithvi stats are in raw HLS DN (×10000) scale.
# Order: B02 (Blue), B03 (Green), B04 (Red), B05 (NIR_Narrow), B06 (SWIR1), B07 (SWIR2).
HLS_MEAN = torch.tensor([0.1087, 0.1342, 0.1433, 0.2734, 0.1958, 0.1363])
HLS_STD  = torch.tensor([0.2248, 0.2179, 0.2178, 0.1850, 0.1242, 0.1049])

# Our B4/B8/B11 channels are stored as raw DN / 10000 (see unet_common.stack_to_tensors).
# That puts them on the same 0-1 reflectance scale as HLS surface reflectance.
# Index in HLS layout where each of our bands goes:
#   B4  -> Red       (idx 2)
#   B8  -> NIR       (idx 3)
#   B11 -> SWIR1     (idx 4)
OUR_S2_TO_HLS_IDX = {0: 2, 1: 3, 2: 4}  # source idx in our stack -> dest idx in HLS layout
HLS_MISSING = (0, 1, 5)  # Blue, Green, SWIR2 -- zero-filled


class S2ToHLSAdapter(nn.Module):
    """Project our 3-band S2 input into a 6-band HLS tensor with Prithvi normalization.

    Input  : (B, 3, H, W)   [B4, B8, B11], 0-1 reflectance scale
    Output : (B, 6, H, W)   normalized HLS layout
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", HLS_MEAN.view(1, 6, 1, 1))
        self.register_buffer("std", HLS_STD.view(1, 6, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 3, f"expected 3 S2 bands, got {x.shape[1]}"
        B, _, H, W = x.shape
        out = x.new_zeros((B, 6, H, W))
        for src, dst in OUR_S2_TO_HLS_IDX.items():
            out[:, dst] = x[:, src]
        return (out - self.mean) / self.std


def select_spectral(stack: torch.Tensor) -> torch.Tensor:
    """Pull [B4, B8, B11] out of our 9-channel stack."""
    return stack[:, [0, 1, 2]]


def select_side(stack: torch.Tensor) -> torch.Tensor:
    """Pull [ETa300m, NDVI, NDMI, FVC, sinDOY, cosDOY] out of our 9-channel stack."""
    return stack[:, [3, 4, 5, 6, 7, 8]]

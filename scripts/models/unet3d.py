"""3D Residual U-Net — ported from fedpod-old scripts/models/."""
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Blocks ───────────────────────────────────────────────────────────────────

_NORM = {'instance': nn.InstanceNorm3d, 'batch': nn.BatchNorm3d}


class _Identity(nn.Module):
    def forward(self, x): return x


class PlainBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3,
                 norm_key='instance', dropout_prob=None):
        super().__init__()
        conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2, bias=True)
        do   = _Identity() if dropout_prob is None else nn.Dropout3d(dropout_prob, inplace=True)
        norm = _NORM[norm_key](out_ch, eps=1e-5, affine=True)
        self.block = nn.Sequential(conv, do, norm, nn.LeakyReLU(inplace=True))

    def forward(self, x): return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3,
                 norm_key='instance', dropout_prob=None):
        super().__init__()
        conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2, bias=True)
        norm = _NORM[norm_key](out_ch, eps=1e-5, affine=True)
        do   = _Identity() if dropout_prob is None else nn.Dropout3d(dropout_prob, inplace=True)
        self.block = nn.Sequential(conv, norm, do, nn.LeakyReLU(inplace=True))

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, bias=True),
                _NORM[norm_key](out_ch, eps=1e-5, affine=True),
            )
        else:
            self.skip = _Identity()

    def forward(self, x):
        return self.skip(x) + self.block(x)


# ── Encoder / Decoder ────────────────────────────────────────────────────────

class UNetEncoder(nn.Module):
    def __init__(self, in_ch, channels, block=PlainBlock, **bkw):
        super().__init__()
        self.levels = nn.ModuleList()
        for l, out_ch in enumerate(channels):
            in_c = in_ch if l == 0 else channels[l - 1]
            stride = 1 if l == 0 else 2
            self.levels.append(nn.Sequential(
                block(in_c, out_ch, stride=stride, **bkw),
                block(out_ch, out_ch, stride=1, **bkw),
            ))

    def forward(self, x, return_skips=False):
        skips = []
        for s in self.levels:
            x = s(x)
            skips.append(x)
        return skips if return_skips else x


class UNetDecoder(nn.Module):
    def __init__(self, out_classes, channels, deep_supervision=False,
                 ds_layer=0, block=PlainBlock, **bkw):
        super().__init__()
        n_up = len(channels) - 1

        self.trans_convs = nn.ModuleList()
        self.levels       = nn.ModuleList()
        for l in range(n_up):
            in_c, out_c = channels[l], channels[l + 1]
            self.trans_convs.append(
                nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2))
            self.levels.append(nn.Sequential(
                block(out_c * 2, out_c, stride=1, **bkw),
                block(out_c, out_c, stride=1, **bkw),
            ))

        self.seg_output = nn.Conv3d(channels[-1], out_classes, 1)

        self.deep_supervision = deep_supervision and ds_layer > 1
        if self.deep_supervision:
            self.ds_idx = list(range(n_up - ds_layer, n_up - 1))
            self.ds = nn.ModuleList()
            for l in range(n_up - 1):
                if l in self.ds_idx:
                    up = channels[l + 1] // channels[-1]
                    self.ds.append(nn.Sequential(
                        nn.Conv3d(channels[l + 1], out_classes, 1),
                        nn.Upsample(scale_factor=up, mode='trilinear', align_corners=False),
                    ))
                else:
                    self.ds.append(None)

    def forward(self, skips):
        skips = skips[::-1]
        x = skips.pop(0)
        ds_outs = []
        for l, feat in enumerate(skips):
            x = self.trans_convs[l](x)
            x = torch.cat([feat, x], dim=1)
            x = self.levels[l](x)
            if self.training and self.deep_supervision and l in self.ds_idx:
                ds_outs.append(self.ds[l](x))
        if self.training and self.deep_supervision:
            return [self.seg_output(x)] + ds_outs[::-1]
        return self.seg_output(x)


class UNet(nn.Module):
    """3D Residual U-Net.

    Args:
        in_ch:      input channels (e.g. 4 for t1/t1ce/t2/flair)
        out_classes: output channels (number of segmentation targets)
        channels:   feature channels per encoder level, e.g. [32, 64, 128, 256]
        block:      PlainBlock or ResidualBlock
        deep_supervision / ds_layer: deep supervision settings
    """

    def __init__(self, in_ch, out_classes, channels,
                 block=ResidualBlock, deep_supervision=False, ds_layer=0,
                 **bkw):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, channels, block=block, **bkw)
        self.decoder = UNetDecoder(out_classes, channels[::-1], block=block,
                                   deep_supervision=deep_supervision,
                                   ds_layer=ds_layer, **bkw)

    def forward(self, x):
        return self.decoder(self.encoder(x, return_skips=True))

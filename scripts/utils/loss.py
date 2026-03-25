"""SoftDice + BCE combined loss — ported from fedpod-old."""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def robust_sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=0.0, max=1.0)


def _sum_tensor(inp, axes):
    for ax in sorted(np.unique(axes).astype(int), reverse=True):
        inp = inp.sum(int(ax))
    return inp


def _get_tp_fp_fn(net_output, gt):
    axes = tuple(range(2, net_output.dim()))
    shp_x, shp_y = net_output.shape, gt.shape

    with torch.no_grad():
        if net_output.dim() != gt.dim():
            gt = gt.view(shp_y[0], 1, *shp_y[1:])
        if net_output.shape == gt.shape:
            y_oh = gt
        else:
            y_oh = torch.zeros_like(net_output)
            y_oh.scatter_(1, gt.long(), 1)

    tp = net_output * y_oh
    fp = net_output * (1 - y_oh)
    fn = (1 - net_output) * y_oh

    tp = _sum_tensor(tp, axes)
    fp = _sum_tensor(fp, axes)
    fn = _sum_tensor(fn, axes)
    return tp, fp, fn


class SoftDiceWithLogitsLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, x, y):
        x = robust_sigmoid(x)
        tp, fp, fn = _get_tp_fp_fn(x, y)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth + 1e-8)
        return 1 - dc  # shape [B, C]


class SoftDiceBCEWithLogitsLoss(nn.Module):
    """BCE + SoftDice combined loss (returns both separately)."""

    def __init__(self, dice_smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dsc = SoftDiceWithLogitsLoss(smooth=dice_smooth)

    def forward(self, net_output: Tensor, target: Tensor):
        bce_loss = self.bce(net_output, target)
        dsc_loss = self.dsc(net_output, target)   # [B, C]
        return bce_loss, dsc_loss

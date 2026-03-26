"""Committee evaluation: score each training case using Stage 1 model.

Runs Stage 1 WT model on institution's train cases and computes per-case
Dice scores. Outputs a priority JSON for use in Stage 2 WeightedRandomSampler.

Usage:
    python scripts/run_committee.py \
        --model_path  states/stage1_local/inst1/R01r00/models/R01r00_last.pth \
        --inst_ids    1 \
        --cases_split experiments/partition2/fets_split.csv \
        --data_root   /data/fets128/trainval \
        --input_channels "[t1,t1ce,t2,flair]" \
        --label_groups   "[[1,2,4]]" \
        --output_path states/stage1_local/inst1/priority.json \
        --priority_low  0.3 \
        --priority_high 0.9
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from dsets.fets import FeTSDataset
from models.unet3d import UNet, ResidualBlock, PlainBlock
from utils.metrics import dice
from utils.misc import load_subjects

_BLOCKS = {'plain': PlainBlock, 'residual': ResidualBlock}


def _parse_list(s):
    s = s.strip()
    if s.startswith('[') and not s.startswith('[['):
        return [x.strip() for x in s[1:-1].split(',') if x.strip()]
    return ast.literal_eval(s)


def compute_weights(scores, lo, hi):
    """Map Dice scores to sample weights.

    Scores in [lo, hi] → weight 1.0 (informative cases)
    Below lo           → linearly ramp up  (possibly noisy, down-weight)
    Above hi           → linearly ramp down (too easy, down-weight)
    """
    weights = np.ones(len(scores), dtype=np.float32)
    for i, s in enumerate(scores):
        if s < lo:
            weights[i] = max(s / lo, 1e-3)
        elif s > hi:
            weights[i] = max((1.0 - s) / (1.0 - hi), 1e-3)
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',     required=True)
    p.add_argument('--inst_ids',       type=int, nargs='+', default=[1])
    p.add_argument('--cases_split',    default='experiments/partition2/fets_split.csv')
    p.add_argument('--data_root',      default='data/fets128/trainval')
    p.add_argument('--input_channels', default='[t1,t1ce,t2,flair]')
    p.add_argument('--label_groups',   default='[[1,2,4]]')
    p.add_argument('--block',          default='residual', choices=['plain', 'residual'])
    p.add_argument('--channels_list',  default='[32,64,128,256]')
    p.add_argument('--output_path',    required=True)
    p.add_argument('--priority_low',   type=float, default=0.3)
    p.add_argument('--priority_high',  type=float, default=0.9)
    p.add_argument('--use_gpu',        type=int,   default=1)
    args = p.parse_args()

    channel_names = _parse_list(args.input_channels)
    label_groups  = ast.literal_eval(args.label_groups)
    channels_list = ast.literal_eval(args.channels_list)
    in_ch         = len(channel_names)
    num_classes   = len(label_groups)

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    # ── load model ───────────────────────────────────────────────────────────
    block = _BLOCKS[args.block]
    model = UNet(in_ch=in_ch, out_classes=num_classes,
                 channels=channels_list, block=block)
    ckpt = torch.load(args.model_path, map_location='cpu')
    sd   = ckpt.get('model', ckpt)
    sd   = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f'Loaded: {args.model_path}  (in={in_ch}, out={num_classes})')

    # ── dataset (train cases only) ───────────────────────────────────────────
    train_cases = load_subjects(args.cases_split, args.inst_ids, 'train')
    ds = FeTSDataset(
        data_root=args.data_root,
        case_names=train_cases,
        channel_names=channel_names,
        label_groups=label_groups,
        mode='val',
        flip_lr=False,
    )

    # ── evaluate each case ───────────────────────────────────────────────────
    scores = {}
    with torch.no_grad():
        for idx in range(len(ds)):
            img, lbl, name, _aff = ds[idx]
            img = img.unsqueeze(0).to(device)
            lbl = lbl.unsqueeze(0).to(device)

            out  = model(img)
            if isinstance(out, list):
                out = out[0]
            pred = (torch.sigmoid(out) > 0.5).float()
            dc   = dice(pred, lbl).mean().item()
            scores[name] = round(dc, 4)

    score_vals = [scores[n] for n in train_cases]
    weights    = compute_weights(score_vals, args.priority_low, args.priority_high)

    # ── save priority JSON ───────────────────────────────────────────────────
    result = {
        'cases':    train_cases,
        'scores':   score_vals,
        'weights':  weights.tolist(),
        'config': {
            'model_path':    args.model_path,
            'priority_low':  args.priority_low,
            'priority_high': args.priority_high,
        },
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    mean_score = float(np.mean(score_vals))
    print(f'Cases: {len(train_cases)}  mean_dice={mean_score:.4f}')
    print(f'Weights  min={weights.min():.4f}  max={weights.max():.4f}  '
          f'mean={weights.mean():.4f}')
    print(f'Priority saved → {out_path}')


if __name__ == '__main__':
    main()

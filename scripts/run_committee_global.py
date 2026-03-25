"""Global committee evaluation: score all institutions' cases and produce a
single globally-ranked priority file for Static Active Learning.

Each institution uses its own Stage 1 model to evaluate its local training
cases. All scores are pooled and sorted globally (descending), so that
top-K% selection reflects global data informativeness rather than
per-institution relative ranking.

Usage:
    python scripts/run_committee_global.py \\
        --inst_models "1:states/stage1_local/inst1/R01r00/models/R01r00_last.pth,\\
                       2:states/stage1_local/inst2/R01r00/models/R01r00_last.pth" \\
        --cases_split experiments/partition2/fets_split.csv \\
        --data_root   data/fets128/trainval \\
        --input_channels "[t1,t1ce,t2,flair]" \\
        --label_groups   "[[1,2,4]]" \\
        --output_path    states/stage1_local/global_priority.json
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


def _parse_inst_models(s):
    """Parse '1:path1,2:path2' → {1: 'path1', 2: 'path2'}"""
    result = {}
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        inst_id, path = token.split(':', 1)
        result[int(inst_id.strip())] = path.strip()
    return result


def _load_model(model_path, in_ch, num_classes, channels_list, block, device):
    model = UNet(in_ch=in_ch, out_classes=num_classes,
                 channels=channels_list, block=block)
    ckpt = torch.load(model_path, map_location='cpu')
    sd   = ckpt.get('model', ckpt)
    sd   = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


def _eval_cases(model, ds, device):
    """Evaluate all cases in ds. Returns {case_name: dice_score}."""
    scores = {}
    with torch.no_grad():
        for idx in range(len(ds)):
            img, lbl, name = ds[idx]
            img = img.unsqueeze(0).to(device)
            lbl = lbl.unsqueeze(0).to(device)
            out = model(img)
            if isinstance(out, list):
                out = out[0]
            pred = (torch.sigmoid(out) > 0.5).float()
            dc = dice(pred, lbl).mean().item()
            scores[name] = round(dc, 4)
    return scores


def main():
    p = argparse.ArgumentParser(
        description='Global committee evaluation for Static Active Learning')
    p.add_argument('--inst_models',    required=True,
                   help='Comma-separated "inst_id:model_path" pairs. '
                        'e.g. "1:states/s1/inst1/R01r00/models/R01r00_last.pth,'
                        '2:states/s1/inst2/R01r00/models/R01r00_last.pth"')
    p.add_argument('--cases_split',    default='experiments/partition2/fets_split.csv')
    p.add_argument('--data_root',      default='data/fets128/trainval')
    p.add_argument('--input_channels', default='[t1,t1ce,t2,flair]')
    p.add_argument('--label_groups',   default='[[1,2,4]]')
    p.add_argument('--block',          default='residual', choices=['plain', 'residual'])
    p.add_argument('--channels_list',  default='[32,64,128,256]')
    p.add_argument('--output_path',    required=True)
    p.add_argument('--use_gpu',        type=int, default=1)
    args = p.parse_args()

    inst_models   = _parse_inst_models(args.inst_models)
    channel_names = _parse_list(args.input_channels)
    label_groups  = ast.literal_eval(args.label_groups)
    channels_list = ast.literal_eval(args.channels_list)
    in_ch         = len(channel_names)
    num_classes   = len(label_groups)
    block         = _BLOCKS[args.block]
    device        = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    print(f'Institutions: {sorted(inst_models.keys())}')
    print(f'Input: {in_ch}ch  Classes: {num_classes}  Device: {device}')

    # ── evaluate each institution ─────────────────────────────────────────────
    inst_results = {}  # {inst_id: {'cases': [...], 'scores': [...]}}

    for inst_id, model_path in sorted(inst_models.items()):
        print(f'\n[Inst {inst_id}] Model: {model_path}')
        model = _load_model(model_path, in_ch, num_classes,
                            channels_list, block, device)

        train_cases = load_subjects(args.cases_split, [inst_id], 'train')
        ds = FeTSDataset(
            data_root=args.data_root,
            case_names=train_cases,
            channel_names=channel_names,
            label_groups=label_groups,
            mode='val',
            flip_lr=False,
        )

        scores = _eval_cases(model, ds, device)
        score_list = [scores[c] for c in train_cases]
        mean_score = float(np.mean(score_list))
        print(f'  Cases: {len(train_cases)}  mean_dice={mean_score:.4f}  '
              f'min={min(score_list):.4f}  max={max(score_list):.4f}')

        inst_results[str(inst_id)] = {
            'cases':  train_cases,
            'scores': score_list,
        }

        del model  # free GPU memory between institutions

    # ── global ranking ────────────────────────────────────────────────────────
    all_cases  = []
    all_scores = []
    for idata in inst_results.values():
        all_cases.extend(idata['cases'])
        all_scores.extend(idata['scores'])

    # sort by score descending (highest score = index 0)
    sorted_pairs = sorted(zip(all_scores, all_cases), reverse=True)
    all_scores_sorted = [s for s, _ in sorted_pairs]
    all_cases_sorted  = [c for _, c in sorted_pairs]

    total = len(all_cases_sorted)
    print(f'\nGlobal pool: {total} cases  '
          f'mean={float(np.mean(all_scores_sorted)):.4f}  '
          f'top-10%≥{all_scores_sorted[max(0, int(total*0.1)-1)]:.4f}  '
          f'top-30%≥{all_scores_sorted[max(0, int(total*0.3)-1)]:.4f}')

    # ── save ──────────────────────────────────────────────────────────────────
    result = {
        'all_cases':  all_cases_sorted,   # score-descending order
        'all_scores': all_scores_sorted,
        'institutions': inst_results,
        'config': {
            'models':         {str(k): v for k, v in inst_models.items()},
            'input_channels': args.input_channels,
            'label_groups':   args.label_groups,
            'data_root':      args.data_root,
            'cases_split':    args.cases_split,
        },
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nGlobal priority saved → {out_path}')


if __name__ == '__main__':
    main()

"""Create and save a randomly initialized model for FL round 0.

All institutions in round 0 load from this shared init.pth so they
start from the same parameter space.

Output:
    states/{agg_name}/init.pth
"""
import sys
import ast
import argparse
import logging
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.unet3d import UNet, ResidualBlock, PlainBlock

_BLOCKS = {'plain': PlainBlock, 'residual': ResidualBlock}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='FedPOD init model creation')
    p.add_argument('--agg_name',      type=str, required=True)
    p.add_argument('--states_root',   type=str, default='states')
    p.add_argument('--block',         type=str, default='residual',
                   choices=['plain', 'residual'])
    p.add_argument('--channels_list', type=str, default='[32,64,128,256]')
    p.add_argument('--input_channels',type=str, default='[t1,t1ce,t2,flair]')
    p.add_argument('--label_groups',  type=str, default='[[1,2,4]]')
    p.add_argument('--seed',          type=int, default=0,
                   help='random seed for reproducible init')
    return p.parse_args(argv)


def main():
    args = parse_args()

    # ── parse architecture params ─────────────────────────────────────────────
    channels   = ast.literal_eval(args.channels_list)
    chan_names = [x.strip() for x in args.input_channels.strip()[1:-1].split(',')]
    label_grps = ast.literal_eval(args.label_groups)
    in_ch      = len(chan_names)
    out_cls    = len(label_grps)

    # ── build and init model ──────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    block = _BLOCKS[args.block]
    model = UNet(in_ch=in_ch, out_classes=out_cls, channels=channels, block=block)

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.states_root) / args.agg_name / 'init.pth'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, out_path)

    logging.basicConfig(format='%(asctime)s  %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO)
    logging.info(f'Init model saved → {out_path}  '
                 f'(in={in_ch}, out={out_cls}, ch={channels}, seed={args.seed})')


if __name__ == '__main__':
    main()

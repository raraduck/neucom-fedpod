"""Argument parser for run_train.py."""
import argparse
import ast


def _parse_list(s: str) -> list:
    """'[t1,t1ce,t2,flair]' → ['t1','t1ce','t2','flair']
       '[wt]' → ['wt']"""
    s = s.strip()
    if s.startswith('[') and not s.startswith('[['):
        inner = s[1:-1]
        return [x.strip() for x in inner.split(',') if x.strip()]
    return ast.literal_eval(s)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='FedPOD training')

    # ── meta ────────────────────────────────────────────────────────────────
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--save_infer', type=int,   default=0)
    p.add_argument('--eval_freq',  type=int,   default=5)
    p.add_argument('--use_gpu',    type=int,   default=1)
    p.add_argument('--job_name',   type=str,   default='test_job')
    p.add_argument('--algorithm',  type=str,   default='fedavg',
                   choices=['fedavg', 'fedprox', 'fedpod'])
    p.add_argument('--mu',         type=float, default=0.001,
                   help='FedProx proximal term weight')

    # ── FL rounds ───────────────────────────────────────────────────────────
    p.add_argument('--rounds',     type=int,   default=1,
                   help='total FL rounds')
    p.add_argument('--round',      type=int,   default=0,
                   help='current FL round (0-indexed)')
    p.add_argument('--epochs',     type=int,   default=30,
                   help='total epochs per round')
    p.add_argument('--epoch',      type=int,   default=0,
                   help='epoch offset (resume from)')

    # ── data ────────────────────────────────────────────────────────────────
    p.add_argument('--data_root',  type=str,   default='data/fets128/trainval',
                   help='path to folder containing case directories '
                        '(e.g. data/fets128/trainval). '
                        'FL institution split is done via --cases_split CSV.')
    p.add_argument('--dataset',    type=str,   default='fets')
    p.add_argument('--cases_split',type=str,   default='experiments/partition2/fets_split.csv')
    p.add_argument('--inst_ids',   type=int,   nargs='+', default=[1],
                   help='Partition_ID values to include from the CSV '
                        '(FL institution filter, not a folder name)')
    p.add_argument('--data_pct',   type=float, default=1.0,
                   help='fraction of training data to use')

    p.add_argument('--input_channels', type=str,
                   default='[t1,t1ce,t2,flair]',
                   help='modality list, e.g. [t1,t1ce,t2,flair]')
    p.add_argument('--label_groups',   type=str,
                   default='[[1,2,4]]',
                   help='label groups, e.g. [[1,2,4]] or [[1],[2],[4]]')
    p.add_argument('--label_names',    type=str,
                   default='[wt]',
                   help='label names, e.g. [wt] or [ncr,ed,et]')
    p.add_argument('--priority_path',  type=str, default='',
                   help='path to priority.json (local) or global_priority.json; '
                        'used by WeightedRandomSampler (select_mode=all) or '
                        'committee case filtering (select_mode=committee)')
    p.add_argument('--select_pct',    type=float, default=1.0,
                   help='fraction of train cases to use for Static AL '
                        '(0 < x <= 1.0). Applied before data_pct.')
    p.add_argument('--select_mode',   type=str, default='all',
                   choices=['all', 'committee', 'random'],
                   help='all: use all cases (or WeightedRandomSampler if priority_path set); '
                        'committee: top select_pct%% by global_priority score; '
                        'random: seed-based random select_pct%% subset')
    p.add_argument('--mask_channels',  type=str, default='[]',
                   help='input channels to binarize (>0) instead of z-score, '
                        'e.g. [seg] for Stage 2 WT mask input')
    p.add_argument('--label_index',    type=str, default='[]',
                   help='original NIfTI label values per output class, '
                        'e.g. [1] for WT-only, [2,1,4] for WT/TC/ET. '
                        'Defaults to [1..num_classes] if omitted.')

    # ── augmentation ────────────────────────────────────────────────────────
    p.add_argument('--zoom',      type=int, default=0,
                   help='legacy flag (unused); resize is now always applied via _FGZoomCenter')
    p.add_argument('--flip_lr',   type=int, default=1)
    p.add_argument('--resize',    type=int, default=128,
                   help='target cubic preprocessing size. '
                        'Raw volumes are zoomed so FG fills this cube, '
                        'then padded/cropped to resize³. '
                        'e.g. 128 (fets128) or 192 (for fets240).')
    p.add_argument('--patch_size',type=int, default=128,
                   help='cubic patch size for training. '
                        'If patch_size == resize: full preprocessed volume is used. '
                        'If patch_size < resize: one FG-guaranteed random crop '
                        'of patch_size³ is extracted per sample per iteration.')

    # ── model ───────────────────────────────────────────────────────────────
    p.add_argument('--weight_path',   type=str,   default='none')
    p.add_argument('--channels_list', type=str,   default='[32,64,128,256]')
    p.add_argument('--block',         type=str,   default='residual',
                   choices=['plain', 'residual'])
    p.add_argument('--deep_supervision', type=int, default=0)
    p.add_argument('--ds_layer',      type=int,   default=1)
    p.add_argument('--kernel_size',   type=int,   default=3)
    p.add_argument('--dropout_prob',  type=float, default=None)
    p.add_argument('--norm',          type=str,   default='instance')

    # ── optimiser ───────────────────────────────────────────────────────────
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--milestones',   type=str,   default='[30]',
                   help='epoch milestones for LR decay, e.g. [20,40]')
    p.add_argument('--lr_gamma',     type=float, default=0.1)
    p.add_argument('--batch_size',   type=int,   default=1)

    args = p.parse_args(argv)

    # ── post-parse conversions ───────────────────────────────────────────────
    args.input_channel_names  = _parse_list(args.input_channels)
    args.mask_channel_names   = _parse_list(args.mask_channels) if args.mask_channels.strip() != '[]' else []
    args.label_groups         = ast.literal_eval(args.label_groups)
    args.label_names         = _parse_list(args.label_names)
    args.channels_list       = ast.literal_eval(args.channels_list)
    args.milestones          = ast.literal_eval(args.milestones)
    args.num_classes         = len(args.label_groups)
    args.in_channels         = len(args.input_channel_names)

    # label_index: list of original NIfTI values per output class
    # e.g. WT-only → [1], three-class WT/TC/ET → [2,1,4]
    # Falls back to [1, 2, ..., num_classes] when not provided or mismatched.
    raw_idx = ast.literal_eval(args.label_index) if args.label_index.strip() else []
    if not raw_idx or len(raw_idx) != args.num_classes:
        raw_idx = list(range(1, args.num_classes + 1))
    args.label_index = raw_idx

    return args

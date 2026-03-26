"""Aggregation entry point for fedpod-new.

Reads _last.pth from each local client job, computes weighted aggregation,
and saves the global model as _agg.pth for the next round.

Usage:
    python scripts/run_aggregation.py \\
        --job_names inst01_job inst02_job inst03_job \\
        --agg_name  global_job \\
        --rounds 3  --round 0  \\
        --algorithm fedpod

Output:
    states/{agg_name}/R{R:02}r{next_round:02}/models/R{R:02}r{next_round:02}_agg.pth
    states/{agg_name}/metrics.json   (running history of pre/post metrics per round)
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from aggregator import aggregate, compute_weights


# ── arg parser ────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='FedPOD aggregation')
    p.add_argument('--job_names',  nargs='+', required=True,
                   help='job_name of each local client (matches training --job_name)')
    p.add_argument('--agg_name',   type=str,  required=True,
                   help='name for the aggregated output job')
    p.add_argument('--rounds',     type=int,  default=3,
                   help='total FL rounds')
    p.add_argument('--round',      type=int,  default=0,
                   help='current round being aggregated (0-indexed)')
    p.add_argument('--algorithm',  type=str,  default='fedavg',
                   choices=['fedavg', 'fedwavg', 'fedprox', 'fedpod', 'fedpid'])
    p.add_argument('--states_root', type=str, default='states',
                   help='root directory for all states')
    return p.parse_args(argv)


# ── logger ────────────────────────────────────────────────────────────────────

def _init_logger(agg_name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(agg_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s  %(message)s', '%H:%M:%S')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / 'aggregation.log')
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _last_pth_path(states_root: Path, job_name: str, rounds: int, round_: int) -> Path:
    return (states_root / job_name
            / f'R{rounds:02d}r{round_:02d}' / 'models'
            / f'R{rounds:02d}r{round_:02d}_last.pth')


def _agg_pth_path(states_root: Path, agg_name: str,
                  rounds: int, next_round: int) -> Path:
    return (states_root / agg_name
            / f'R{rounds:02d}r{next_round:02d}' / 'models'
            / f'R{rounds:02d}r{next_round:02d}_agg.pth')


# ── JSON metrics history ──────────────────────────────────────────────────────

def _load_json_history(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json_history(path: Path, history: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


# ── selection CSV merge ───────────────────────────────────────────────────────

def _merge_selection_csvs(job_names: list, root: Path,
                          rounds: int, round_: int,
                          agg_name: str, logger: logging.Logger):
    """Collect per-institution fets_split.csv and accumulate into one file.

    A single states/{agg_name}/fets_split.csv is maintained across rounds.
    Each call adds/updates the R{round_} column for the current round.
    Previous round columns (R0, R1, ...) are preserved as-is.

    R{round_} values: 1=selected, 0=not selected, NaN=val/test row.
    """
    r_col    = f'R{round_}'
    out_path = root / agg_name / 'fets_split.csv'
    base_df  = None

    # ── collect R{round_} marks from each institution ─────────────────────────
    for jname in job_names:
        csv_path = (root / jname
                    / f'R{rounds:02d}r{round_:02d}' / 'fets_split.csv')
        if not csv_path.exists():
            logger.warning(f'  [CSV merge] not found: {csv_path}')
            continue

        inst_df = pd.read_csv(csv_path)

        if base_df is None:
            # Seed from existing accumulated file, or first institution CSV
            if out_path.exists():
                base_df = pd.read_csv(out_path)
                # Ensure the new round column exists
                if r_col not in base_df.columns:
                    base_df[r_col] = float('nan')
            else:
                base_df = inst_df.copy()
                base_df[r_col] = float('nan')

        # Copy this institution's marks (0/1) into base
        marked = (inst_df[inst_df[r_col].notna()]
                  .set_index('Subject_ID')[r_col])
        base_df = base_df.set_index('Subject_ID')
        base_df.loc[marked.index, r_col] = marked
        base_df = base_df.reset_index()

    if base_df is None:

        logger.warning('  [CSV merge] No per-institution CSVs found — skipping.')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_df[r_col] = base_df[r_col].astype('Int64')
    base_df.to_csv(out_path, index=False)

    selected     = int((base_df[r_col] == 1).sum())
    marked_total = int(base_df[r_col].notna().sum())
    logger.info(
        f'  Selection accumulated → {out_path}  '
        f'[{r_col}] selected {selected} / marked {marked_total}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    root  = Path(args.states_root)
    round_ = args.round
    next_  = round_ + 1
    algo   = args.algorithm

    log_dir = Path('logs') / args.agg_name
    logger  = _init_logger(args.agg_name, log_dir)
    logger.info('=' * 60)
    logger.info(f'Aggregation  Round: {round_}/{args.rounds}  Algorithm: {algo.upper()}')
    logger.info(f'Clients: {args.job_names}')

    # ── load local _last.pth files ────────────────────────────────────────────
    local_states = []
    for jname in args.job_names:
        path = _last_pth_path(root, jname, args.rounds, round_)
        assert path.exists(), f'Missing checkpoint: {path}'
        state = torch.load(path, map_location='cpu')
        local_states.append(state)
        logger.info(f'  Loaded {path}  '
                    f'P={state["P"]}  I={state["I"]:.4f}  D={state["D"]:.4f}')

    # ── load JSON history (needed for fedpid) ─────────────────────────────────
    json_path    = root / args.agg_name / 'metrics.json'
    json_history = _load_json_history(json_path)

    # ── update history with current round metrics ──────────────────────────────
    round_record = {}
    for jname, state in zip(args.job_names, local_states):
        round_record[jname] = {
            'pre':  state['pre_metrics'],
            'post': state['post_metrics'],
            'P':    state['P'],
            'I':    state['I'],
            'D':    state['D'],
        }
    json_history[str(round_)] = round_record
    _save_json_history(json_path, json_history)
    logger.info(f'  Metrics history → {json_path}')

    # ── compute weights ───────────────────────────────────────────────────────
    weights = compute_weights(algo, local_states, json_history, round_)
    models  = [s['model'] for s in local_states]

    for jname, state, w in zip(args.job_names, local_states, weights):
        tag = ' (pre-val only)' if state['P'] == 0 else ''
        logger.info(
            f'  [{jname}]  P={state["P"]}  I={state["I"]:.4f}  '
            f'D={state["D"]:.4f}  W={w:.4f}{tag}')

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg_model = aggregate(weights, models)
    logger.info(f'  Weight sum = {sum(weights):.6f}')

    # ── save aggregated model ─────────────────────────────────────────────────
    out_path = _agg_pth_path(root, args.agg_name, args.rounds, next_)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': agg_model}, out_path)
    logger.info(f'  Aggregated → {out_path}')

    # ── summary ──────────────────────────────────────────────────────────────
    mean_pre_loss  = sum(s['pre_metrics']['total']  for s in local_states) / len(local_states)
    mean_post_loss = sum(s['post_metrics']['total'] for s in local_states) / len(local_states)
    mean_pre_dice  = sum(s['pre_metrics']['dice']   for s in local_states) / len(local_states)
    mean_post_dice = sum(s['post_metrics']['dice']  for s in local_states) / len(local_states)
    logger.info(
        f'  Mean  pre: loss={mean_pre_loss:.4f}  dice={mean_pre_dice:.4f}')
    logger.info(
        f'  Mean post: loss={mean_post_loss:.4f}  dice={mean_post_dice:.4f}')
    logger.info(f'Done. Next round model: {out_path}')

    # ── TensorBoard logging ───────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = Path('runs') / args.agg_name
        writer = SummaryWriter(str(tb_dir))
        step = round_  # x-axis = round index

        # per-institution metrics
        for jname, state in zip(args.job_names, local_states):
            tag = jname.split('/')[-1]  # e.g. inst1, inst2
            pm  = state['pre_metrics']
            qm  = state['post_metrics']
            writer.add_scalar(f'inst/{tag}/pre_loss',  pm['total'], step)
            writer.add_scalar(f'inst/{tag}/pre_dice',  pm['dice'],  step)
            writer.add_scalar(f'inst/{tag}/post_loss', qm['total'], step)
            writer.add_scalar(f'inst/{tag}/post_dice', qm['dice'],  step)

        # averages
        writer.add_scalar('avg/pre_loss',  mean_pre_loss,  step)
        writer.add_scalar('avg/pre_dice',  mean_pre_dice,  step)
        writer.add_scalar('avg/post_loss', mean_post_loss, step)
        writer.add_scalar('avg/post_dice', mean_post_dice, step)

        writer.close()
        logger.info(f'  TensorBoard → {tb_dir}')
    except ImportError:
        pass

    # ── merge per-institution selection CSVs ──────────────────────────────────
    _merge_selection_csvs(args.job_names, root,
                          args.rounds, round_, args.agg_name, logger)


if __name__ == '__main__':
    main()

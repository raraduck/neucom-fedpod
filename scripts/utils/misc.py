"""Utility helpers."""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_subjects(split_path: str, inst_ids: list[int],
                  split_type: str) -> list[str]:
    """Return Subject_IDs for the given partition(s) and train/val split.

    Args:
        split_path: path to fets_split.csv
        inst_ids:   list of Partition_ID values (e.g. [1])
        split_type: 'train' or 'val'
    """
    df = pd.read_csv(split_path)
    df = df[df['Partition_ID'].isin(inst_ids)]
    df = df[df['TrainOrVal'] == split_type]
    return df['Subject_ID'].tolist()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / max(self.count, 1)


def init_logger(job_name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s  %(message)s', '%H:%M:%S')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / 'train.log')
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger

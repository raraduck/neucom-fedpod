"""FeTS dataset loader for fedpod-new."""
import json
import os

import nibabel as nib
import numpy as np
import torch
import monai.transforms as T
from torch.utils.data import Dataset, WeightedRandomSampler


# ── MONAI custom transform ────────────────────────────────────────────────────

class _RobustZScore(T.MapTransform):
    """Robust z-score normalisation using non-zero foreground voxels."""
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            vol = d[key]
            mask = vol > 0
            if mask.sum() == 0:
                continue
            lo = float(np.percentile(vol[mask], 0.2))
            hi = float(np.percentile(vol[mask], 99.8))
            vol = np.clip(vol, lo, hi)
            fg = vol[mask]
            vol = (vol - fg.mean()) / (fg.std() + 1e-8)
            d[key] = vol
        return d


class _Binarize(T.MapTransform):
    """Convert a mask channel to binary (> 0). Used for seg input in Stage 2."""
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (d[key] > 0).astype(np.float32)
        return d


class _ToMultiChannel(T.MapTransform):
    """Convert integer label map to multi-channel binary masks."""
    def __init__(self, keys, label_groups):
        super().__init__(keys)
        self.groups = label_groups

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lmap = np.squeeze(d[key], axis=0)          # (D, H, W)
            channels = []
            for group in self.groups:
                ch = np.zeros_like(lmap, dtype=np.float32)
                for lbl in group:
                    ch = np.logical_or(ch, lmap == lbl).astype(np.float32)
                channels.append(ch)
            d[key] = np.stack(channels, axis=0)         # (C, D, H, W)
        return d


def _nib_load(path: str):
    proxy = nib.load(path)
    data  = proxy.get_fdata()
    proxy.uncache()
    return np.expand_dims(data, axis=0).astype(np.float32)  # (1, D, H, W)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeTSDataset(Dataset):
    """FeTS dataset for fets128 (128³ volumes, no resize needed).

    Args:
        data_root:     path to the folder containing case directories,
                       e.g. 'data/fets128/trainval'
                       FL institution split is handled by the CSV (Partition_ID),
                       not by folder structure.
        case_names:    list of case IDs (filtered from CSV by Partition_ID)
        channel_names: list of modality names, e.g. ['t1','t1ce','t2','flair']
        label_groups:  label groups for segmentation, e.g. [[1,2,4]]
        mode:          'train' or 'val'
        flip_lr:       apply random L/R flip during training
    """

    def make_sampler(self) -> WeightedRandomSampler | None:
        """Return WeightedRandomSampler if priority_path was set, else None."""
        if self._weights is None:
            return None
        return WeightedRandomSampler(
            weights=self._weights,
            num_samples=len(self._weights),
            replacement=True,
        )

    def __init__(self, data_root: str, case_names: list,
                 channel_names: list, label_groups: list,
                 mode: str = 'train', flip_lr: bool = True,
                 mask_channels: list = None,
                 priority_path: str = None):
        self.data_root     = data_root
        self.case_names    = case_names
        self.channel_names = channel_names
        self.label_groups  = label_groups
        self.mode          = mode.lower()
        self.flip_lr       = flip_lr and self.mode == 'train'

        # ── priority weights (Committee-guided sampling) ─────────────────────
        self._weights = None
        if priority_path and os.path.exists(priority_path):
            with open(priority_path) as f:
                pdata = json.load(f)
            # align weights to case_names order
            w_map = dict(zip(pdata['cases'], pdata['weights']))
            self._weights = torch.tensor(
                [w_map.get(n, 1.0) for n in case_names], dtype=torch.float32)

        mask_ch  = mask_channels or []
        mri_ch   = [c for c in channel_names if c not in mask_ch]

        # base: normalise + stack → image tensor; convert label
        base_keys = channel_names + ['label']
        tfms = [T.EnsureTyped(keys=base_keys)]
        if mri_ch:
            tfms.append(_RobustZScore(keys=mri_ch))
        if mask_ch:
            tfms.append(_Binarize(keys=mask_ch))
        tfms += [
            T.ConcatItemsd(keys=channel_names, name='image', dim=0),
            T.DeleteItemsd(keys=channel_names),
            _ToMultiChannel(keys=['label'], label_groups=label_groups),
        ]
        self._base = T.Compose(tfms)

        # aug: random flips (train only)
        aug_list = []
        if self.flip_lr:
            aug_list += [
                T.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                T.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
                T.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            ]
        self._aug = T.Compose(aug_list) if aug_list else None

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        name     = self.case_names[index]
        case_dir = os.path.join(self.data_root, name)

        # load modalities
        d = {}
        for mod in self.channel_names:
            d[mod] = _nib_load(os.path.join(case_dir, f'{name}_{mod}.nii.gz'))

        # load label (sub.nii.gz = NCR/ED/ET)
        d['label'] = _nib_load(os.path.join(case_dir, f'{name}_sub.nii.gz'))

        d = self._base(d)
        if self._aug is not None:
            d = self._aug(d)

        image = torch.as_tensor(np.array(d['image']), dtype=torch.float32)
        label = torch.as_tensor(np.array(d['label']), dtype=torch.float32)
        return image, label, name

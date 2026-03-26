"""FeTS dataset for fedpod-new.

Preprocessing (zoom=1) — applied to full volume:
  CropForegroundd  → Spacingd (FG fills resize)  → SpatialPadd (resize³)
  ConvertToMultiChannel (label)

Augmentation (train) — applied to patch:
  RandCropByPosNegLabeld  (patch_size³, FG-guaranteed when patch_size < resize)
  RobustZScoreNormalization  (MRI channels)
  _Binarize                  (mask channels)
  ConcatItemsd → 'image'
  RandFlipd × 3  |  RandRotated  |  RandGaussianNoised
  RandGaussianSmoothd  |  RandAdjustContrastd

Validation — applied to full preprocessed volume:
  RobustZScoreNormalization  |  _Binarize  |  ConcatItemsd → 'image'
"""
import json
import os

import nibabel as nib
import numpy as np
import torch
import monai.transforms as T
from monai.transforms.transform import MapTransform
from torch.utils.data import Dataset, WeightedRandomSampler

MASKS = ['seg', 'ref', 'sub', 'label']


# ── Custom transforms ─────────────────────────────────────────────────────────

class RobustZScoreNormalization(MapTransform):
    """Robust z-score using non-zero foreground voxels (fedpod-old style)."""
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0
            if not mask.any():
                continue
            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)
            d[key][mask & (d[key] < lower)] = float(lower)
            d[key][mask & (d[key] > upper)] = float(upper)
            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()
        return d


class _Binarize(MapTransform):
    """Convert mask channel to binary (>0). Used for seg input in Stage 2."""
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (np.asarray(d[key]) > 0).astype(np.float32)
        return d


class ConvertToMultiChannel(MapTransform):
    """Integer label map → multi-channel binary masks."""
    def __init__(self, keys, label_groups):
        super().__init__(keys)
        self.groups = label_groups

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lmap     = np.squeeze(np.asarray(d[key]), axis=0)   # (D, H, W)
            channels = []
            for group in self.groups:
                ch = np.zeros_like(lmap, dtype=np.float32)
                for lbl in group:
                    ch = np.logical_or(ch, lmap == lbl).astype(np.float32)
                channels.append(ch)
            d[key] = np.stack(channels, axis=0)                 # (C, D, H, W)
        return d


def _nib_load(path: str) -> np.ndarray:
    proxy = nib.load(path)
    data  = proxy.get_fdata()
    proxy.uncache()
    return np.expand_dims(data, axis=0).astype(np.float32)   # (1, D, H, W)


def _nib_affine(path: str) -> np.ndarray:
    """Load only the affine matrix from a NIfTI file."""
    return nib.load(path).affine


# ── Transform builders (fedpod-old style) ─────────────────────────────────────

def _get_base_transforms(channel_names, label_groups, mask_ch, resize, zoom):
    """Spatial preprocessing transforms (applied to full volume).
    zoom=True : CropForeground → Spacingd → SpatialPad → resize³
    zoom=False: only EnsureTyped + label conversion (data already at target size)
    """
    selected_keys = channel_names + ['label']
    mri_ch = [c for c in channel_names if c not in MASKS and c not in mask_ch]

    base = [T.EnsureTyped(keys=selected_keys)]

    if zoom:
        def _calc_shape(data):
            data['_crop_shape'] = data[channel_names[0]].shape[1:]
            return data

        def _apply_spacing(data):
            max_sz = max(data['_crop_shape'])
            ratio  = max_sz / resize
            modes  = ['nearest' if k in MASKS or k in mask_ch else 'bilinear'
                      for k in selected_keys]
            return T.Spacingd(keys=selected_keys, pixdim=(ratio,)*3,
                              mode=modes)(data)

        base += [
            T.CropForegroundd(keys=selected_keys, source_key=channel_names[0],
                              margin=(10, 10, 10), k_divisible=[1, 1, 1]),
            T.Lambda(func=_calc_shape),
            T.Lambda(func=_apply_spacing),
            T.SpatialPadd(keys=selected_keys,
                          spatial_size=(resize, resize, resize), mode='constant'),
        ]

    base += [ConvertToMultiChannel(keys=['label'], label_groups=label_groups)]
    return base


def _get_aug_transforms(channel_names, mask_ch, patch_size, resize, flip_lr):
    """Augmentation transforms applied to patch (train only)."""
    mri_ch = [c for c in channel_names if c not in MASKS and c not in mask_ch]
    all_ch_keys = channel_names + ['label']

    aug = []

    # FG-guaranteed patch crop (skip when patch_size == resize → use full volume)
    if patch_size < resize:
        aug += [
            T.RandCropByPosNegLabeld(
                keys=all_ch_keys,
                label_key='label',
                spatial_size=[patch_size] * 3,
                pos=1.0, neg=0.0,
                num_samples=1,
            ),
        ]

    # Normalize on patch
    if mri_ch:
        aug += [RobustZScoreNormalization(keys=mri_ch)]
    if mask_ch:
        aug += [_Binarize(keys=mask_ch)]

    # Concat → 'image'
    aug += [
        T.EnsureTyped(keys=channel_names),   # unify types after Binarize/ZScore
        T.ConcatItemsd(keys=channel_names, name='image', dim=0),
        T.DeleteItemsd(keys=channel_names),
    ]

    # Spatial augmentation
    if flip_lr:
        aug += [
            T.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        ]
    aug += [
        T.RandRotated(keys=['image', 'label'], prob=0.5,
                      mode=['bilinear', 'nearest'],
                      range_x=0.3, range_y=0.3, range_z=0.3),
    ]

    # Intensity augmentation
    aug += [
        T.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        T.RandGaussianSmoothd(keys='image', prob=0.15,
                              sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5),
                              sigma_z=(0.5, 1.5)),
        T.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),
        T.EnsureTyped(keys=['image', 'label']),
    ]
    return aug


def _get_val_transforms(channel_names, mask_ch):
    """Normalization + concat only (no random ops)."""
    mri_ch = [c for c in channel_names if c not in MASKS and c not in mask_ch]
    tfms = []
    if mri_ch:
        tfms += [RobustZScoreNormalization(keys=mri_ch)]
    if mask_ch:
        tfms += [_Binarize(keys=mask_ch)]
    tfms += [
        T.EnsureTyped(keys=channel_names),   # unify types after Binarize/ZScore
        T.ConcatItemsd(keys=channel_names, name='image', dim=0),
        T.DeleteItemsd(keys=channel_names),
        T.EnsureTyped(keys=['image', 'label']),
    ]
    return tfms


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeTSDataset(Dataset):
    """FeTS dataset with configurable preprocessing and patch-based training.

    Args:
        resize:     target cubic size after FG-zoom preprocessing.
                    Used only when zoom=True.
        patch_size: cubic patch size.
                    patch_size == resize → full preprocessed volume (no crop).
                    patch_size  < resize → FG-guaranteed random patch per iter.
        zoom:       True  → CropForeground + Spacingd + SpatialPad (for raw data
                            such as fets240 240×240×155 or any non-standard size).
                    False → skip spatial resize (data already at target size,
                            e.g. fets128 128³).
    """

    def make_sampler(self) -> WeightedRandomSampler | None:
        if self._weights is None:
            return None
        return WeightedRandomSampler(
            weights=self._weights,
            num_samples=len(self._weights),
            replacement=True,
        )

    def __init__(
        self,
        data_root: str,
        case_names: list,
        channel_names: list,
        label_groups: list,
        mode: str = 'train',
        flip_lr: bool = True,
        mask_channels: list = None,
        priority_path: str = None,
        resize: int = 128,
        patch_size: int = 128,
        zoom: bool = False,
    ):
        self.data_root     = data_root
        self.case_names    = case_names
        self.channel_names = channel_names
        self.mode          = mode.lower()
        self.flip_lr       = flip_lr and self.mode == 'train'

        mask_ch = mask_channels or []

        # ── priority weights ─────────────────────────────────────────────────
        self._weights = None
        if priority_path and os.path.exists(priority_path):
            with open(priority_path) as f:
                pdata = json.load(f)
            w_map = dict(zip(pdata['cases'], pdata['weights']))
            self._weights = torch.tensor(
                [w_map.get(n, 1.0) for n in case_names], dtype=torch.float32)

        # ── build transform pipelines ─────────────────────────────────────────
        base_tfms = _get_base_transforms(
            channel_names, label_groups, mask_ch, resize, zoom)
        self._base = T.Compose(base_tfms)

        if self.mode == 'train':
            aug_tfms = _get_aug_transforms(
                channel_names, mask_ch, patch_size, resize, self.flip_lr)
            self._aug = T.Compose(aug_tfms)
        else:
            val_tfms = _get_val_transforms(channel_names, mask_ch)
            self._val = T.Compose(val_tfms)

        self._use_patch = (patch_size < resize) and (self.mode == 'train')

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        name     = self.case_names[index]
        case_dir = os.path.join(self.data_root, name)

        # Load raw NIfTI; capture affine from first channel for later NIfTI saving
        d = {}
        affine = None
        for mod in self.channel_names:
            path = os.path.join(case_dir, f'{name}_{mod}.nii.gz')
            d[mod] = _nib_load(path)
            if affine is None:
                affine = _nib_affine(path)
        d['label'] = _nib_load(os.path.join(case_dir, f'{name}_sub.nii.gz'))

        # Spatial preprocessing (full volume)
        d = self._base(d)

        if self.mode == 'train':
            d = self._aug(d)
            # RandCropByPosNegLabeld returns a list when num_samples=1
            if isinstance(d, list):
                d = d[0]
        else:
            d = self._val(d)

        image  = torch.as_tensor(np.asarray(d['image']), dtype=torch.float32)
        label  = torch.as_tensor(np.asarray(d['label']), dtype=torch.float32)
        affine = torch.as_tensor(affine, dtype=torch.float64)  # (4, 4)
        return image, label, name, affine

"""Global committee evaluation: score all cases using ALL institution models
and produce a globally-ranked priority file for Static Active Learning.

Uncertainty metrics:
  BALD (Mutual Information):
    MI = H[E_k(p)] - E_k[H(p)]
       = Predictive entropy  -  Expected entropy
       = Total uncertainty   -  Aleatoric uncertainty
       = Epistemic (inter-model) uncertainty

  roi_bald:
    Same as BALD, but averaged only within the predicted WT foreground ROI
    (p_mean > roi_threshold).  Background voxels are excluded so that
    large empty regions do not dilute the epistemic signal.

  texture_bald  (recommended):
    roi_bald × (1 + λ × texture_complexity)

    texture_complexity is the mean of min-max normalised texture features
    extracted from each input modality within the WT ROI:
      • intensity entropy  (histogram-based, 32 bins)
      • intensity std-dev
      • ROI volume ratio   (tumour load)

    Cases with both high model-disagreement AND high structural diversity
    in the ROI receive the highest scores, correcting the tendency of pure
    BALD to cluster selections around a narrow texture subspace.

Ranking convention: all_cases sorted DESCENDING by chosen score_key
  → top-K% = most uncertain / most texture-diverse = best for AL

Usage:
    python scripts/run_committee_global.py \\
        --inst_models "1:states/s1/inst1/R01r00/models/R01r00_last.pth,\\
                       2:states/s1/inst2/R01r00/models/R01r00_last.pth" \\
        --cases_split experiments/partition2/fets_split.csv \\
        --data_root   data/fets128/trainval \\
        --input_channels "[t1,t1ce,t2,flair]" \\
        --label_groups   "[[1,2,4]]" \\
        --output_path    states/stage1_local/global_priority.json \\
        --score_key      texture_bald \\
        --diversity_lambda 0.5
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
_EPS = 1e-6


# ── uncertainty computation ───────────────────────────────────────────────────

def compute_uncertainty(prob_maps: list, roi_threshold: float = 0.3) -> dict:
    """Compute BALD (MI), ROI-BALD, and Variance from K probability maps.

    Args:
        prob_maps:     list of K tensors, each shape (1, C, D, H, W), values in [0,1]
        roi_threshold: probability threshold to define the WT foreground ROI

    Returns dict of scalar scores:
        bald_mi:    Epistemic uncertainty averaged over the full volume
        roi_bald:   Epistemic uncertainty averaged only within predicted WT ROI
        aleatoric:  Expected entropy per model (data ambiguity)
        predictive: Total predictive entropy of committee mean
        variance:   Prediction variance across models
        p_mean:     (1, C, D, H, W) mean probability map (returned for texture extraction)
    """
    # p_stack: (K, C, D, H, W)
    p_stack = torch.cat(prob_maps, dim=0)

    p_mean = p_stack.mean(dim=0, keepdim=True)   # (1, C, D, H, W)

    # Predictive entropy: H[E_k[p]]
    H_pred = _binary_entropy(p_mean)             # (1, C, D, H, W)

    # Expected entropy: E_k[H[p_k]]
    H_each = _binary_entropy(p_stack)            # (K, C, D, H, W)
    H_exp  = H_each.mean(dim=0, keepdim=True)    # (1, C, D, H, W)

    # BALD mutual information (epistemic)
    bald = (H_pred - H_exp).clamp(min=0.0)       # (1, C, D, H, W)

    # Variance across models
    variance = p_stack.var(dim=0, unbiased=False, keepdim=True)

    # ROI-masked BALD: restrict to predicted foreground to avoid background dilution
    # ROI = any channel where committee mean probability exceeds threshold
    roi_mask = (p_mean.squeeze(0) > roi_threshold).any(dim=0)  # (D, H, W)
    bald_vol = bald.squeeze(0).mean(dim=0)                      # (D, H, W)
    if roi_mask.sum() > 10:
        roi_bald = bald_vol[roi_mask].mean().item()
    else:
        roi_bald = bald.mean().item()   # fallback when no ROI detected

    return {
        'bald_mi':    bald.mean().item(),
        'roi_bald':   roi_bald,
        'aleatoric':  H_exp.mean().item(),
        'predictive': H_pred.mean().item(),
        'variance':   variance.mean().item(),
        '_p_mean':    p_mean,   # internal; stripped before saving
    }


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Element-wise binary entropy: -p*log(p) - (1-p)*log(1-p)."""
    return -(p * torch.log(p + _EPS)
             + (1 - p) * torch.log(1 - p + _EPS))


# ── ROI texture extraction ────────────────────────────────────────────────────

def extract_roi_texture(img: torch.Tensor, p_mean: torch.Tensor,
                        roi_threshold: float = 0.3) -> np.ndarray:
    """Extract texture descriptors from input image within predicted WT ROI.

    For each input modality:
      • intensity entropy  (histogram-based, 32 bins)
      • intensity std-dev  within ROI
    Plus:
      • ROI volume ratio   (tumour load, fraction of total voxels)

    These features characterise whether a case has rich / heterogeneous
    tissue within the tumour region — complementing the model-uncertainty
    signal from BALD, which does not encode image appearance.

    Args:
        img:           (1, C, D, H, W) input image (z-scored)
        p_mean:        (1, C_cls, D, H, W) mean committee probability map
        roi_threshold: threshold to binarise the soft ROI mask

    Returns:
        float32 array of length 2*C + 1
    """
    C = img.shape[1]

    p_sq = p_mean.squeeze(0)                          # (C_cls, D, H, W) or (D,H,W)
    if p_sq.ndim == 4:
        mask = (p_sq > roi_threshold).any(dim=0)      # (D, H, W)
    else:
        mask = (p_sq > roi_threshold)                 # (D, H, W)

    mask_np = mask.cpu().numpy()
    n_roi   = int(mask_np.sum())

    feats = []
    for c in range(C):
        ch_np = img[0, c].cpu().numpy()               # (D, H, W)

        if n_roi >= 8:
            roi_vals = ch_np[mask_np].astype(np.float64)
            # Intensity histogram entropy
            hist, _ = np.histogram(roi_vals, bins=32)
            hist     = hist.astype(np.float64) + 1e-8
            hist    /= hist.sum()
            entropy  = float(-np.sum(hist * np.log(hist)))
            # Intensity std
            std = float(roi_vals.std())
        else:
            entropy = 0.0
            std     = 0.0

        feats.append(entropy)
        feats.append(std)

    # Tumour load
    vol_ratio = n_roi / float(mask_np.size)
    feats.append(vol_ratio)

    return np.array(feats, dtype=np.float32)


# ── texture-aware combined score ──────────────────────────────────────────────

def compute_texture_bald(case_metrics: dict, texture_feats: dict,
                         lambda_div: float = 0.5) -> None:
    """Add 'texture_bald' to each entry in case_metrics (in-place).

    texture_bald[i] = roi_bald[i] × (1 + λ × texture_complexity[i])

    texture_complexity is the mean of min-max normalised texture features,
    rewarding cases that are uncertain AND texturally diverse.

    Args:
        case_metrics:  {case_name: metric_dict}  (modified in-place)
        texture_feats: {case_name: float32 array}
        lambda_div:    diversity weight (0 = pure roi_bald, 1 = equal weight)
    """
    cases = list(case_metrics.keys())
    if not cases:
        return

    feat_matrix = np.stack([
        texture_feats.get(c, np.zeros(1)) for c in cases
    ])                                                  # (N, D)

    # Normalise each feature dimension to [0, 1] across the dataset
    feat_min  = feat_matrix.min(axis=0, keepdims=True)
    feat_max  = feat_matrix.max(axis=0, keepdims=True)
    feat_norm = (feat_matrix - feat_min) / (feat_max - feat_min + 1e-8)

    tex_complexity = feat_norm.mean(axis=1)             # (N,)

    for i, c in enumerate(cases):
        rb = case_metrics[c].get('roi_bald', case_metrics[c].get('bald_mi', 0.0))
        case_metrics[c]['texture_bald'] = float(
            rb * (1.0 + lambda_div * tex_complexity[i]))


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Global committee uncertainty evaluation for Static Active Learning')
    p.add_argument('--inst_models',    required=True,
                   help='Comma-separated "inst_id:model_path" pairs. '
                        'e.g. "1:path1.pth,2:path2.pth"')
    p.add_argument('--cases_split',    default='experiments/partition2/fets_split.csv')
    p.add_argument('--data_root',      default='data/fets128/trainval')
    p.add_argument('--input_channels', default='[t1,t1ce,t2,flair]')
    p.add_argument('--label_groups',   default='[[1,2,4]]')
    p.add_argument('--block',          default='residual', choices=['plain', 'residual'])
    p.add_argument('--channels_list',  default='[32,64,128,256]')
    p.add_argument('--output_path',    required=True)
    p.add_argument('--resize',         type=int, default=128,
                   help='target preprocessing cube size (must match training resize)')
    p.add_argument('--score_key',      default='texture_bald',
                   choices=['bald_mi', 'roi_bald', 'variance', 'mean_dice', 'texture_bald'],
                   help='Metric used for ranking cases (sorted descending). '
                        'texture_bald: roi_bald weighted by ROI texture complexity (recommended); '
                        'roi_bald: BALD within predicted WT foreground only; '
                        'bald_mi: BALD over full volume; '
                        'variance: prediction variance; '
                        'mean_dice: committee mean Dice vs GT')
    p.add_argument('--diversity_lambda', type=float, default=0.5,
                   help='Weight of texture diversity in texture_bald score. '
                        '0 = pure roi_bald, 1 = equal weight to texture complexity.')
    p.add_argument('--roi_threshold',  type=float, default=0.3,
                   help='Probability threshold to define the WT ROI mask '
                        'for roi_bald and texture feature extraction.')
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

    print(f'Institutions : {sorted(inst_models.keys())}')
    print(f'Input        : {in_ch}ch  Classes: {num_classes}  Device: {device}')
    print(f'Score key    : {args.score_key}  (λ_div={args.diversity_lambda})')
    print(f'ROI threshold: {args.roi_threshold}')
    print(f'Cross-eval   : ALL {len(inst_models)} models on ALL cases')

    # ── collect all cases across institutions ─────────────────────────────────
    all_inst_cases = {}  # {inst_id: [case_names]}
    for inst_id in sorted(inst_models.keys()):
        all_inst_cases[inst_id] = load_subjects(
            args.cases_split, [inst_id], 'train')

    all_cases_flat = []
    case_to_inst   = {}
    for inst_id, cases in all_inst_cases.items():
        for c in cases:
            all_cases_flat.append(c)
            case_to_inst[c] = inst_id

    print(f'Total cases  : {len(all_cases_flat)} '
          f'({", ".join(f"inst{k}:{len(v)}" for k,v in all_inst_cases.items())})')

    # ── load all models ───────────────────────────────────────────────────────
    print('\nLoading models...')
    models = {}
    for inst_id, model_path in sorted(inst_models.items()):
        models[inst_id] = _load_model(
            model_path, in_ch, num_classes, channels_list, block, device)
        print(f'  [Inst {inst_id}] Loaded: {model_path}')

    # ── build a single dataset covering all cases ─────────────────────────────
    ds = FeTSDataset(
        data_root=args.data_root,
        case_names=all_cases_flat,
        channel_names=channel_names,
        label_groups=label_groups,
        mode='val',
        flip_lr=False,
        resize=args.resize,
        patch_size=args.resize,   # always full volume for evaluation
    )

    # ── evaluate each case with ALL models ────────────────────────────────────
    print(f'\nEvaluating {len(ds)} cases × {len(models)} models...')
    case_metrics  = {}   # {case_name: metric_dict}
    texture_feats = {}   # {case_name: float32 feature array}

    K          = len(models)
    model_list = [models[k] for k in sorted(models.keys())]

    with torch.no_grad():
        for idx in range(len(ds)):
            img, lbl, name, _aff = ds[idx]
            img = img.unsqueeze(0).to(device)   # (1, C, D, H, W)
            lbl = lbl.unsqueeze(0).to(device)   # (1, cls, D, H, W)

            # collect probability maps from all K models
            prob_maps   = []
            dice_scores = []
            for model in model_list:
                out = model(img)
                if isinstance(out, list):
                    out = out[0]
                prob = torch.sigmoid(out)         # (1, C, D, H, W)
                prob_maps.append(prob)

                if args.score_key == 'mean_dice':
                    pred_bin = (prob > 0.5).float()
                    dc = dice(pred_bin, lbl).mean().item()
                    dice_scores.append(dc)

            # uncertainty (returns p_mean for texture extraction)
            unc   = compute_uncertainty(prob_maps, roi_threshold=args.roi_threshold)
            p_mean = unc.pop('_p_mean')          # (1, C, D, H, W); not saved to JSON
            unc['mean_dice'] = float(np.mean(dice_scores)) if dice_scores else 0.0
            case_metrics[name] = unc

            # texture features within predicted WT ROI
            texture_feats[name] = extract_roi_texture(
                img.cpu(), p_mean.cpu(), roi_threshold=args.roi_threshold)

            if (idx + 1) % 50 == 0:
                print(f'  {idx+1}/{len(ds)}  '
                      f'bald={unc["bald_mi"]:.4f}  '
                      f'roi_bald={unc["roi_bald"]:.4f}  '
                      f'aleatoric={unc["aleatoric"]:.4f}  '
                      f'variance={unc["variance"]:.4f}')

    # ── compute texture-aware combined score ──────────────────────────────────
    compute_texture_bald(case_metrics, texture_feats, lambda_div=args.diversity_lambda)

    # ── store texture features in metrics (for analysis) ─────────────────────
    feat_len = len(channel_names) * 2 + 1
    feat_names = []
    for ch in channel_names:
        feat_names += [f'tex_entropy_{ch}', f'tex_std_{ch}']
    feat_names.append('tex_vol_ratio')

    for name in all_cases_flat:
        fv = texture_feats.get(name, np.zeros(feat_len))
        for fn, fval in zip(feat_names, fv):
            case_metrics[name][fn] = round(float(fval), 6)

    # ── global ranking ────────────────────────────────────────────────────────
    score_values = [case_metrics[c][args.score_key] for c in all_cases_flat]

    sorted_pairs      = sorted(zip(score_values, all_cases_flat), reverse=True)
    all_scores_sorted = [s for s, _ in sorted_pairs]
    all_cases_sorted  = [c for _, c in sorted_pairs]

    total = len(all_cases_sorted)
    def _pct_threshold(pct):
        idx = max(0, int(total * pct) - 1)
        return all_scores_sorted[idx]

    print(f'\n{"─"*60}')
    print(f'Score key : {args.score_key}  (sorted descending)')
    print(f'Total     : {total} cases')
    print(f'Mean      : {float(np.mean(all_scores_sorted)):.4f}')
    print(f'Top-10%   ≥ {_pct_threshold(0.10):.4f}')
    print(f'Top-20%   ≥ {_pct_threshold(0.20):.4f}')
    print(f'Top-30%   ≥ {_pct_threshold(0.30):.4f}')

    # per-institution breakdown
    inst_results = {}
    for inst_id, cases in all_inst_cases.items():
        inst_scores = [case_metrics[c][args.score_key] for c in cases]
        inst_results[str(inst_id)] = {
            'cases':  cases,
            'scores': [round(s, 6) for s in inst_scores],
        }
        print(f'Inst {inst_id} ({len(cases)} cases) '
              f'mean={float(np.mean(inst_scores)):.4f}  '
              f'min={min(inst_scores):.4f}  max={max(inst_scores):.4f}')

    # ── save ──────────────────────────────────────────────────────────────────
    result = {
        'all_cases':    all_cases_sorted,
        'all_scores':   [round(s, 6) for s in all_scores_sorted],
        'score_key':    args.score_key,
        'diversity_lambda': args.diversity_lambda,
        'metrics':      {c: {k: round(v, 6) for k, v in m.items()}
                         for c, m in case_metrics.items()},
        'institutions': inst_results,
        'config': {
            'models':         {str(k): v for k, v in inst_models.items()},
            'input_channels': args.input_channels,
            'label_groups':   args.label_groups,
            'data_root':      args.data_root,
            'cases_split':    args.cases_split,
            'num_models':     K,
            'roi_threshold':  args.roi_threshold,
        },
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nGlobal priority saved → {out_path}')


if __name__ == '__main__':
    main()

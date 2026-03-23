#!/usr/bin/env python3
"""
Segmentation metadata analysis for FeTS2022 dataset.

For each brain image, computes:
  - Image metadata  : shape, voxel spacing, volume
  - seg (WT)        : Whole Tumor — binary mask (label 1)
  - sub subregions  : NCR (label 1), ED (label 2), ET (label 4)
                      and WT = NCR+ED+ET combined

Results are saved to:
    analysis/results/{data}/partition{N}/seg_stats.csv

Usage (run from fedpod-new/):
    python analysis/run_seg_stats.py --data fets240 --partition 1
    python analysis/run_seg_stats.py --data fets128 --partition 1
    python analysis/run_seg_stats.py --data fets240 fets128 --partition 1 2
"""

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS  = PROJECT_ROOT / "experiments"
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"

# 멀티프로세싱 worker 에서 공유할 전역 경로
_DATA_ROOT: Path = None


def _init_worker(data_root_str: str) -> None:
    global _DATA_ROOT
    _DATA_ROOT = Path(data_root_str)


# ── 통계 계산 ─────────────────────────────────────────────────────────────

def mask_stats(mask: np.ndarray, affine: np.ndarray, prefix: str) -> dict:
    voxel_vol_mm3 = float(np.abs(np.linalg.det(affine[:3, :3])))

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return {
            f"{prefix}_voxels":  0,
            f"{prefix}_vol_mm3": 0.0,
            f"{prefix}_cx_vox":  float("nan"),
            f"{prefix}_cy_vox":  float("nan"),
            f"{prefix}_cz_vox":  float("nan"),
            f"{prefix}_cx_mm":   float("nan"),
            f"{prefix}_cy_mm":   float("nan"),
            f"{prefix}_cz_mm":   float("nan"),
            f"{prefix}_bbox_x":  0,
            f"{prefix}_bbox_y":  0,
            f"{prefix}_bbox_z":  0,
        }

    centroid_vox = coords.mean(axis=0)
    centroid_mm  = nib.affines.apply_affine(affine, centroid_vox)
    bbox         = coords.max(axis=0) - coords.min(axis=0) + 1

    return {
        f"{prefix}_voxels":  int(len(coords)),
        f"{prefix}_vol_mm3": round(len(coords) * voxel_vol_mm3, 2),
        f"{prefix}_cx_vox":  round(float(centroid_vox[0]), 2),
        f"{prefix}_cy_vox":  round(float(centroid_vox[1]), 2),
        f"{prefix}_cz_vox":  round(float(centroid_vox[2]), 2),
        f"{prefix}_cx_mm":   round(float(centroid_mm[0]),  2),
        f"{prefix}_cy_mm":   round(float(centroid_mm[1]),  2),
        f"{prefix}_cz_mm":   round(float(centroid_mm[2]),  2),
        f"{prefix}_bbox_x":  int(bbox[0]),
        f"{prefix}_bbox_y":  int(bbox[1]),
        f"{prefix}_bbox_z":  int(bbox[2]),
    }


def analyze_case(case_id: str) -> dict | None:
    case_dir = _DATA_ROOT / case_id
    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    sub_path = case_dir / f"{case_id}_sub.nii.gz"

    if not seg_path.exists():
        return None

    seg_nii       = nib.load(str(seg_path))
    affine        = seg_nii.affine
    seg           = np.asarray(seg_nii.dataobj, dtype=np.uint8)
    voxel_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    row: dict = {
        "Subject_ID":   case_id,
        "dim_x":        seg.shape[0],
        "dim_y":        seg.shape[1],
        "dim_z":        seg.shape[2],
        "spacing_x_mm": round(float(voxel_spacing[0]), 4),
        "spacing_y_mm": round(float(voxel_spacing[1]), 4),
        "spacing_z_mm": round(float(voxel_spacing[2]), 4),
        "seg_labels":   str(sorted(np.unique(seg).tolist())),
    }

    row.update(mask_stats(seg == 1, affine, "wt"))

    if sub_path.exists():
        sub = np.asarray(nib.load(str(sub_path)).dataobj, dtype=np.uint8)
        row["sub_labels"] = str(sorted(np.unique(sub).tolist()))
        row.update(mask_stats(sub == 1, affine, "ncr"))
        row.update(mask_stats(sub == 2, affine, "ed"))
        row.update(mask_stats(sub == 4, affine, "et"))
        row.update(mask_stats(sub > 0,  affine, "sub_wt"))
    else:
        row["sub_labels"] = ""

    return row


# ── 파티션 단위 실행 ──────────────────────────────────────────────────────

def run_partition(data: str, partition: int) -> None:
    data_root = PROJECT_ROOT / "data" / data / "inst_01"
    split_csv = EXPERIMENTS / f"partition{partition}" / "fets_split.csv"
    out_dir   = RESULTS_ROOT / data / f"partition{partition}"
    out_csv   = out_dir / "seg_stats.csv"

    if not split_csv.exists():
        print(f"[ERROR] {split_csv} not found", file=sys.stderr)
        return
    if not data_root.exists():
        print(f"[ERROR] {data_root} not found", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    df_split = pd.read_csv(split_csv)[["Subject_ID", "Partition_ID", "TrainOrVal"]]
    case_ids = df_split["Subject_ID"].tolist()
    total    = len(case_ids)
    workers  = min(cpu_count(), 8)

    print(f"\n=== {data} / partition{partition} ({total} cases, {workers} workers) ===")

    results = {}
    with Pool(workers,
              initializer=_init_worker,
              initargs=(str(data_root),)) as pool:
        for i, (case_id, result) in enumerate(
            zip(case_ids, pool.imap(analyze_case, case_ids)), 1
        ):
            results[case_id] = result
            if i % 200 == 0 or i == total:
                print(f"  [{i:4d}/{total}] done")

    rows, failed = [], []
    for _, meta in df_split.iterrows():
        result = results.get(meta["Subject_ID"])
        if result is None:
            failed.append(meta["Subject_ID"])
            continue
        result["Partition_ID"] = meta["Partition_ID"]
        result["TrainOrVal"]   = meta["TrainOrVal"]
        rows.append(result)

    df_out     = pd.DataFrame(rows)
    front_cols = ["Subject_ID", "Partition_ID", "TrainOrVal"]
    rest_cols  = [c for c in df_out.columns if c not in front_cols]
    df_out     = df_out[front_cols + rest_cols]

    df_out.to_csv(out_csv, index=False)
    print(f"  → saved {len(df_out)} rows to {out_csv.relative_to(PROJECT_ROOT)}")

    if failed:
        print(f"  [WARN] missing ({len(failed)}): {failed[:5]}"
              + ("..." if len(failed) > 5 else ""))


# ── 요약 통계 출력 ─────────────────────────────────────────────────────────

def print_summary(data: str, partition: int) -> None:
    csv_path = RESULTS_ROOT / data / f"partition{partition}" / "seg_stats.csv"
    if not csv_path.exists():
        return

    df       = pd.read_csv(csv_path)
    vol_cols = [c for c in df.columns if c.endswith("_vol_mm3")]
    shape    = "×".join(str(df[f"dim_{a}"].iloc[0]) for a in ["x", "y", "z"])

    print(f"\n── {data} / partition{partition} 요약 ({shape} vox) ──")
    print(f"  cases: {len(df)}  "
          f"(train={len(df[df.TrainOrVal=='train'])}, val={len(df[df.TrainOrVal=='val'])})")
    for col in vol_cols:
        region  = col.replace("_vol_mm3", "").upper()
        nonzero = df[col].dropna()
        nonzero = nonzero[nonzero > 0]
        print(f"  {region:8s}  mean={nonzero.mean():9.1f} mm³  "
              f"std={nonzero.std():8.1f}  "
              f"min={nonzero.min():7.1f}  max={nonzero.max():9.1f}  "
              f"empty={len(df)-len(nonzero)}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FeTS2022 seg/sub 메타데이터 분석 → CSV 저장"
    )
    parser.add_argument(
        "--data", nargs="+", default=["fets240"],
        help="분석할 데이터 소스 (예: fets240 fets128)"
    )
    parser.add_argument(
        "--partition", type=int, nargs="+", choices=[1, 2], default=[1, 2],
        help="분석할 파티션 번호 (기본값: 1 2)"
    )
    args = parser.parse_args()

    for data in args.data:
        for p in args.partition:
            run_partition(data, p)
            print_summary(data, p)

    print("\nDone.")


if __name__ == "__main__":
    main()

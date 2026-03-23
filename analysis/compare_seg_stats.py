#!/usr/bin/env python3
"""
fets240 vs fets128 분석 결과 비교.

비교 항목:
  - vol_mm3  : 볼륨 오차 (절대 / 상대)
  - centroid : mm 공간 centroid 거리
  - bbox     : bounding box 크기 차이

결과:
  analysis/results/compare/{partition}/compare_stats.csv   ← 케이스별 차이
  analysis/results/compare/{partition}/outliers.csv        ← 임계값 초과 케이스

Usage (run from fedpod-new/):
    python analysis/compare_seg_stats.py --partition 1
    python analysis/compare_seg_stats.py --partition 1 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"

# 상대 오차 경고 임계값 (%)
VOL_THRESHOLD_PCT = 5.0
# centroid 거리 경고 임계값 (mm)
CENTROID_THRESHOLD_MM = 5.0


def compare_partition(partition: int) -> None:
    p240 = RESULTS_ROOT / "fets240" / f"partition{partition}" / "seg_stats.csv"
    p128 = RESULTS_ROOT / "fets128" / f"partition{partition}" / "seg_stats.csv"

    if not p240.exists() or not p128.exists():
        print(f"[ERROR] CSV not found for partition{partition}. "
              f"Run run_seg_stats.py first.", file=sys.stderr)
        return

    df240 = pd.read_csv(p240).set_index("Subject_ID")
    df128 = pd.read_csv(p128).set_index("Subject_ID")

    # 공통 케이스만 비교
    common = df240.index.intersection(df128.index)
    df240  = df240.loc[common]
    df128  = df128.loc[common]

    print(f"\n=== partition{partition}: {len(common)} cases ===")

    # ── 비교할 region 목록 ────────────────────────────────────────────────
    regions = ["wt", "ncr", "ed", "et", "sub_wt"]
    rows    = []

    for case_id in common:
        r240 = df240.loc[case_id]
        r128 = df128.loc[case_id]
        row  = {"Subject_ID": case_id,
                "Partition_ID": r240["Partition_ID"],
                "TrainOrVal":   r240["TrainOrVal"]}

        for reg in regions:
            vol240 = float(r240.get(f"{reg}_vol_mm3", 0) or 0)
            vol128 = float(r128.get(f"{reg}_vol_mm3", 0) or 0)

            abs_diff = vol128 - vol240
            rel_diff = (abs_diff / vol240 * 100) if vol240 > 0 else float("nan")

            # centroid 거리 (mm 공간)
            cx240 = np.array([r240.get(f"{reg}_cx_mm", np.nan),
                               r240.get(f"{reg}_cy_mm", np.nan),
                               r240.get(f"{reg}_cz_mm", np.nan)], dtype=float)
            cx128 = np.array([r128.get(f"{reg}_cx_mm", np.nan),
                               r128.get(f"{reg}_cy_mm", np.nan),
                               r128.get(f"{reg}_cz_mm", np.nan)], dtype=float)
            if np.any(np.isnan(cx240)) or np.any(np.isnan(cx128)):
                centroid_dist = float("nan")
            else:
                centroid_dist = float(np.linalg.norm(cx240 - cx128))

            row[f"{reg}_vol240"]       = round(vol240, 2)
            row[f"{reg}_vol128"]       = round(vol128, 2)
            row[f"{reg}_vol_abs_diff"] = round(abs_diff, 2)
            row[f"{reg}_vol_rel_pct"]  = round(rel_diff, 3) if not np.isnan(rel_diff) else float("nan")
            row[f"{reg}_centroid_mm"]  = round(centroid_dist, 3) if not np.isnan(centroid_dist) else float("nan")

        rows.append(row)

    df_cmp = pd.DataFrame(rows)

    # ── 요약 출력 ────────────────────────────────────────────────────────
    print(f"\n{'Region':<8}  {'vol_rel_pct(mean)':>18}  "
          f"{'vol_rel_pct(max)':>17}  {'centroid_mm(mean)':>18}  {'centroid_mm(max)':>17}")
    print("-" * 90)
    for reg in regions:
        rel = df_cmp[f"{reg}_vol_rel_pct"].abs().dropna()
        cen = df_cmp[f"{reg}_centroid_mm"].dropna()
        if len(rel) == 0:
            continue
        print(f"{reg.upper():<8}  {rel.mean():>17.3f}%  {rel.max():>16.3f}%  "
              f"{cen.mean():>17.3f}mm  {cen.max():>16.3f}mm")

    # ── outlier 케이스 ────────────────────────────────────────────────────
    outlier_mask = pd.Series(False, index=df_cmp.index)
    for reg in regions:
        rel_col = f"{reg}_vol_rel_pct"
        cen_col = f"{reg}_centroid_mm"
        outlier_mask |= (df_cmp[rel_col].abs() > VOL_THRESHOLD_PCT)
        outlier_mask |= (df_cmp[cen_col] > CENTROID_THRESHOLD_MM)

    df_out = df_cmp[outlier_mask].copy()
    print(f"\n임계값 초과 케이스 (vol>{VOL_THRESHOLD_PCT}% or centroid>{CENTROID_THRESHOLD_MM}mm): "
          f"{len(df_out)} / {len(df_cmp)}")

    # ── 저장 ─────────────────────────────────────────────────────────────
    out_dir = RESULTS_ROOT / "compare" / f"partition{partition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmp_path     = out_dir / "compare_stats.csv"
    outlier_path = out_dir / "outliers.csv"

    df_cmp.to_csv(cmp_path, index=False)
    df_out.to_csv(outlier_path, index=False)

    print(f"  → {cmp_path.relative_to(PROJECT_ROOT)}")
    print(f"  → {outlier_path.relative_to(PROJECT_ROOT)}  ({len(df_out)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="fets240 vs fets128 seg_stats 비교"
    )
    parser.add_argument(
        "--partition", type=int, nargs="+", choices=[1, 2], default=[1],
        help="비교할 파티션 번호 (기본값: 1)"
    )
    args = parser.parse_args()

    for p in args.partition:
        compare_partition(p)

    print("\nDone.")


if __name__ == "__main__":
    main()

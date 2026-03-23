#!/usr/bin/env python3
"""
MNI 공간 WT/subregion 마스크 → Harvard-Oxford Atlas 매핑.

전제: preproc/flirt-sample-job 으로 이미 MNI 정합된 seg / sub 파일이
      data/fets128-mni-reg/{case_id}/ 에 존재해야 함.

출력:
  analysis/results/atlas/{case_id}_atlas.csv   ← region 별 overlap
  analysis/results/atlas/summary.csv           ← 케이스별 top-lobe / laterality

Usage (run from fedpod-new/):
    python analysis/run_atlas_mapping.py
    python analysis/run_atlas_mapping.py --cases FeTS2022_00000 FeTS2022_00002
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REG_ROOT     = PROJECT_ROOT / "data" / "fets128-mni-reg"
ATLAS_DIR    = PROJECT_ROOT / "data" / "atlases"
RESULTS_DIR  = PROJECT_ROOT / "analysis" / "results" / "atlas"

# ── Harvard-Oxford 피질 로브 그룹핑 (label index 0-based) ──────────────────
LOBE_MAP: dict[str, list[int]] = {
    "Frontal": [
        0,   # Frontal Pole
        2,   # Superior Frontal Gyrus
        3,   # Middle Frontal Gyrus
        4,   # Inferior Frontal Gyrus, pars triangularis
        5,   # Inferior Frontal Gyrus, pars opercularis
        6,   # Precentral Gyrus
        24,  # Frontal Medial Cortex
        25,  # Juxtapositional Lobule (Supplementary Motor)
        26,  # Subcallosal Cortex
        27,  # Paracingulate Gyrus
        28,  # Cingulate Gyrus, anterior
        32,  # Frontal Orbital Cortex
        40,  # Frontal Operculum Cortex
        41,  # Central Opercular Cortex
    ],
    "Temporal": [
        7,   # Temporal Pole
        8,   # Superior Temporal Gyrus, anterior
        9,   # Superior Temporal Gyrus, posterior
        10,  # Middle Temporal Gyrus, anterior
        11,  # Middle Temporal Gyrus, posterior
        12,  # Middle Temporal Gyrus, temporooccipital
        13,  # Inferior Temporal Gyrus, anterior
        14,  # Inferior Temporal Gyrus, posterior
        15,  # Inferior Temporal Gyrus, temporooccipital
        36,  # Temporal Fusiform Cortex, anterior
        37,  # Temporal Fusiform Cortex, posterior
        38,  # Temporal Occipital Fusiform Cortex
        43,  # Planum Polare
        44,  # Heschl's Gyrus
        45,  # Planum Temporale
    ],
    "Parietal": [
        16,  # Postcentral Gyrus
        17,  # Superior Parietal Lobule
        18,  # Supramarginal Gyrus, anterior
        19,  # Supramarginal Gyrus, posterior
        20,  # Angular Gyrus
        29,  # Cingulate Gyrus, posterior
        30,  # Precuneous Cortex
        42,  # Parietal Operculum Cortex
    ],
    "Occipital": [
        21,  # Lateral Occipital Cortex, superior
        22,  # Lateral Occipital Cortex, inferior
        23,  # Intracalcarine Cortex
        31,  # Cuneal Cortex
        35,  # Lingual Gyrus
        39,  # Occipital Fusiform Gyrus
        46,  # Supracalcarine Cortex
        47,  # Occipital Pole
    ],
    "Insula": [1],  # Insular Cortex
}

# index → lobe 역매핑
INDEX_TO_LOBE: dict[int, str] = {
    idx: lobe
    for lobe, indices in LOBE_MAP.items()
    for idx in indices
}


# ── atlas 로드 ─────────────────────────────────────────────────────────────

def load_atlas() -> dict:
    """Harvard-Oxford cortical + subcortical atlas 로드."""
    cort_nii  = nib.load(str(ATLAS_DIR / "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"))
    sub_nii   = nib.load(str(ATLAS_DIR / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"))

    cort_arr  = np.asarray(cort_nii.dataobj, dtype=np.uint8)
    sub_arr   = np.asarray(sub_nii.dataobj,  dtype=np.uint8)

    def parse_xml(path: Path) -> dict[int, str]:
        root   = ET.parse(path).getroot()
        return {int(l.get("index")): l.text.strip() for l in root.iter("label")}

    cort_labels = parse_xml(ATLAS_DIR / "HarvardOxford-Cortical.xml")
    sub_labels  = parse_xml(ATLAS_DIR / "HarvardOxford-Subcortical.xml")

    return {
        "cort_arr":    cort_arr,
        "sub_arr":     sub_arr,
        "cort_labels": cort_labels,   # 0-based index → name
        "sub_labels":  sub_labels,
        "affine":      cort_nii.affine,
    }


# ── 케이스 분석 ───────────────────────────────────────────────────────────

def analyze_case(case_id: str, atlas: dict) -> dict | None:
    case_dir = REG_ROOT / case_id
    seg_path = case_dir / f"{case_id}_seg.nii.gz"

    if not seg_path.exists():
        print(f"  [skip] {case_id}: seg not found")
        return None

    seg_arr = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.uint8)
    wt_mask = seg_arr == 1
    n_wt    = int(wt_mask.sum())

    if n_wt == 0:
        print(f"  [skip] {case_id}: WT 없음")
        return None

    cort_arr    = atlas["cort_arr"]
    sub_arr     = atlas["sub_arr"]
    cort_labels = atlas["cort_labels"]
    sub_labels  = atlas["sub_labels"]
    affine      = atlas["affine"]

    # ── laterality (MNI x 좌표 기반) ─────────────────────────────────────
    wt_vox      = np.argwhere(wt_mask)
    ctr_vox     = wt_vox.mean(axis=0)
    ctr_mm      = nib.affines.apply_affine(affine, ctr_vox)
    # MNI 좌표: affine x=-1*voxel+90 → x_mm > 0 = Right, x_mm < 0 = Left
    lat_x       = float(ctr_mm[0])
    laterality  = "Right" if lat_x > 0 else "Left"

    # ── subcortical 명시적 laterality 확인 (Left/Right 라벨 포함) ─────────
    sub_vals    = sub_arr[wt_mask]
    sub_nonzero = sub_vals[sub_vals > 0]
    sub_left_vox  = int(np.isin(sub_nonzero - 1,
                                [i for i,n in sub_labels.items() if "Left"  in n]).sum())
    sub_right_vox = int(np.isin(sub_nonzero - 1,
                                [i for i,n in sub_labels.items() if "Right" in n]).sum())

    # ── cortical 피질 region overlap ─────────────────────────────────────
    cort_vals   = cort_arr[wt_mask]          # atlas value = label_index + 1
    region_rows = []

    for idx, name in cort_labels.items():
        atlas_val = idx + 1
        overlap   = int((cort_vals == atlas_val).sum())
        if overlap == 0:
            continue
        lobe = INDEX_TO_LOBE.get(idx, "Other")
        region_rows.append({
            "Subject_ID": case_id,
            "atlas":      "cortical",
            "region_idx": idx,
            "region":     name,
            "lobe":       lobe,
            "overlap_vox": overlap,
            "overlap_pct": round(overlap / n_wt * 100, 2),
        })

    # subcortical
    for idx, name in sub_labels.items():
        atlas_val = idx + 1
        overlap   = int((sub_vals == atlas_val).sum())
        if overlap == 0:
            continue
        region_rows.append({
            "Subject_ID": case_id,
            "atlas":      "subcortical",
            "region_idx": idx,
            "region":     name,
            "lobe":       "Subcortical",
            "overlap_vox": overlap,
            "overlap_pct": round(overlap / n_wt * 100, 2),
        })

    df_regions = pd.DataFrame(region_rows).sort_values("overlap_pct", ascending=False)

    # ── 로브 집계 ────────────────────────────────────────────────────────
    lobe_sum = (
        df_regions[df_regions["atlas"] == "cortical"]
        .groupby("lobe")["overlap_pct"].sum()
        .sort_values(ascending=False)
    )
    top_lobe     = lobe_sum.index[0]  if len(lobe_sum) else "Unknown"
    top_lobe_pct = float(lobe_sum.iloc[0]) if len(lobe_sum) else 0.0

    # top-3 region
    top3 = df_regions.head(3)[["region", "overlap_pct"]].values.tolist()

    summary = {
        "Subject_ID":     case_id,
        "WT_voxels_mni":  n_wt,
        "centroid_x_mm":  round(lat_x, 1),
        "centroid_y_mm":  round(float(ctr_mm[1]), 1),
        "centroid_z_mm":  round(float(ctr_mm[2]), 1),
        "laterality":     laterality,
        "sub_left_vox":   sub_left_vox,
        "sub_right_vox":  sub_right_vox,
        "top_lobe":       top_lobe,
        "top_lobe_pct":   round(top_lobe_pct, 1),
        "top1_region":    top3[0][0] if len(top3) > 0 else "",
        "top1_pct":       top3[0][1] if len(top3) > 0 else 0.0,
        "top2_region":    top3[1][0] if len(top3) > 1 else "",
        "top2_pct":       top3[1][1] if len(top3) > 1 else 0.0,
        "top3_region":    top3[2][0] if len(top3) > 2 else "",
        "top3_pct":       top3[2][1] if len(top3) > 2 else 0.0,
    }

    return {"summary": summary, "regions": df_regions}


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MNI-registered WT mask → Harvard-Oxford atlas mapping"
    )
    parser.add_argument("--cases", nargs="+", default=None,
                        help="처리할 케이스 ID (기본값: fets128-mni-reg 전체)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    atlas  = load_atlas()
    print(f"Atlas loaded: cortical {len(atlas['cort_labels'])} regions, "
          f"subcortical {len(atlas['sub_labels'])} regions")

    cases  = args.cases if args.cases else sorted(
        p.name for p in REG_ROOT.iterdir() if p.is_dir()
    )
    print(f"Cases: {len(cases)}\n")

    summaries = []
    for case_id in cases:
        print(f"[{case_id}]")
        result = analyze_case(case_id, atlas)
        if result is None:
            continue

        s   = result["summary"]
        df_r = result["regions"]

        # 케이스별 region CSV 저장
        df_r.to_csv(RESULTS_DIR / f"{case_id}_atlas.csv", index=False)

        # 콘솔 출력
        print(f"  WT voxels : {s['WT_voxels_mni']:,}")
        print(f"  centroid  : x={s['centroid_x_mm']:+.1f}  "
              f"y={s['centroid_y_mm']:+.1f}  z={s['centroid_z_mm']:+.1f}  mm")
        print(f"  laterality: {s['laterality']}")
        print(f"  top lobe  : {s['top_lobe']}  ({s['top_lobe_pct']:.1f}%)")
        print(f"  top regions:")
        for _, row in df_r.head(5).iterrows():
            print(f"    {row['overlap_pct']:5.1f}%  {row['region']}  [{row['lobe']}]")
        print()

        summaries.append(s)

    # 전체 요약 CSV 저장
    df_summary = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"→ {summary_path.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()

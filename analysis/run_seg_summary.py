#!/usr/bin/env python3
"""
원본 fets240 seg / sub 마스크에서 WT·NCR·ED·ET의 물리적 수치를 추출.

출력:
  analysis/results/seg_summary.csv  ← 케이스별 볼륨·중심좌표·바운딩박스·비율

Usage (run from fedpod-new/):
    python analysis/run_seg_summary.py
    python analysis/run_seg_summary.py --cases FeTS2022_00000 FeTS2022_00002
"""

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.affines as naff
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data" / "fets240" / "inst_01"
RESULTS_DIR  = PROJECT_ROOT / "analysis" / "results"

# sub 파일 라벨 정의 (BraTS convention)
LABEL_NCR = 1
LABEL_ED  = 2
LABEL_ET  = 4


def region_stats(mask: np.ndarray, affine: np.ndarray, vox_vol: float) -> dict:
    """마스크 1개 영역의 볼륨·중심·바운딩박스 반환. mask가 비어 있으면 None."""
    vox = int(mask.sum())
    if vox == 0:
        return {
            "vol_mm3": 0, "vol_cc": 0.0,
            "cx_mm": None, "cy_mm": None, "cz_mm": None,
            "bbox_x_min": None, "bbox_x_max": None,
            "bbox_y_min": None, "bbox_y_max": None,
            "bbox_z_min": None, "bbox_z_max": None,
            "bbox_span_x": None, "bbox_span_y": None, "bbox_span_z": None,
            "bbox_max_dim": None,
        }

    coords = np.argwhere(mask)            # (N, 3) voxel indices
    ctr_vox = coords.mean(axis=0)
    ctr_mm  = naff.apply_affine(affine, ctr_vox)

    bb_min = coords.min(axis=0)
    bb_max = coords.max(axis=0)
    bb_min_mm = naff.apply_affine(affine, bb_min.astype(float))
    bb_max_mm = naff.apply_affine(affine, bb_max.astype(float))

    # 축별 span은 mm 절댓값 (affine 부호 무관)
    span = np.abs(bb_max_mm - bb_min_mm)

    return {
        "vol_mm3":    round(vox * vox_vol),
        "vol_cc":     round(vox * vox_vol / 1000, 2),
        "cx_mm":      round(float(ctr_mm[0]), 1),
        "cy_mm":      round(float(ctr_mm[1]), 1),
        "cz_mm":      round(float(ctr_mm[2]), 1),
        "bbox_x_min": int(bb_min[0]), "bbox_x_max": int(bb_max[0]),
        "bbox_y_min": int(bb_min[1]), "bbox_y_max": int(bb_max[1]),
        "bbox_z_min": int(bb_min[2]), "bbox_z_max": int(bb_max[2]),
        "bbox_span_x": round(float(span[0]), 1),
        "bbox_span_y": round(float(span[1]), 1),
        "bbox_span_z": round(float(span[2]), 1),
        "bbox_max_dim": round(float(span.max()), 1),
    }


def analyze_case(case_id: str) -> dict | None:
    case_dir = DATA_ROOT / case_id
    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    sub_path = case_dir / f"{case_id}_sub.nii.gz"

    if not seg_path.exists() or not sub_path.exists():
        print(f"  [skip] {case_id}: seg/sub not found")
        return None

    seg_nii = nib.load(str(seg_path))
    sub_nii = nib.load(str(sub_path))

    affine  = seg_nii.affine
    zooms   = seg_nii.header.get_zooms()
    vox_vol = float(zooms[0] * zooms[1] * zooms[2])  # mm³ per voxel

    seg_arr = np.asarray(seg_nii.dataobj, dtype=np.uint8)
    sub_arr = np.asarray(sub_nii.dataobj, dtype=np.uint8)

    wt_mask  = seg_arr == 1
    ncr_mask = sub_arr == LABEL_NCR
    ed_mask  = sub_arr == LABEL_ED
    et_mask  = sub_arr == LABEL_ET
    tc_mask  = ncr_mask | et_mask        # Tumor Core = NCR + ET

    wt  = region_stats(wt_mask,  affine, vox_vol)
    ncr = region_stats(ncr_mask, affine, vox_vol)
    ed  = region_stats(ed_mask,  affine, vox_vol)
    et  = region_stats(et_mask,  affine, vox_vol)
    tc  = region_stats(tc_mask,  affine, vox_vol)

    wt_vol = wt["vol_mm3"]
    safe   = lambda v: round(v / wt_vol * 100, 1) if wt_vol > 0 else 0.0

    # 중심 간 거리 (NCR↔ET: enhancing rim vs necrosis 분리도)
    def dist(a, b):
        if None in (a["cx_mm"], b["cx_mm"]):
            return None
        return round(float(np.linalg.norm([
            a["cx_mm"] - b["cx_mm"],
            a["cy_mm"] - b["cy_mm"],
            a["cz_mm"] - b["cz_mm"],
        ])), 1)

    row = {"Subject_ID": case_id, "vox_vol_mm3": round(vox_vol, 4)}

    # WT
    for k, v in wt.items():
        row[f"WT_{k}"] = v

    # NCR / ED / ET / TC — 볼륨+centroid만 (bbox는 WT로 충분)
    for name, st in [("NCR", ncr), ("ED", ed), ("ET", et), ("TC", tc)]:
        row[f"{name}_vol_mm3"] = st["vol_mm3"]
        row[f"{name}_vol_cc"]  = st["vol_cc"]
        row[f"{name}_cx_mm"]   = st["cx_mm"]
        row[f"{name}_cy_mm"]   = st["cy_mm"]
        row[f"{name}_cz_mm"]   = st["cz_mm"]

    # 비율 (% of WT)
    row["NCR_pct"] = safe(ncr["vol_mm3"])
    row["ED_pct"]  = safe(ed["vol_mm3"])
    row["ET_pct"]  = safe(et["vol_mm3"])
    row["TC_pct"]  = safe(tc["vol_mm3"])

    # 중심 간 거리
    row["dist_NCR_ET_mm"] = dist(ncr, et)
    row["dist_WT_NCR_mm"] = dist(wt, ncr)
    row["dist_WT_ET_mm"]  = dist(wt, et)

    # Laterality: 이미지 x축 중심(vox=120 → mm=-120)을 기준으로 판단
    # affine x=-1*vox+0 → vox 증가 = 해부학적 Left 방향(LAS)
    # WT centroid mm_x < center_mm_x  → vox_x > center → LEFT
    # WT centroid mm_x > center_mm_x  → vox_x < center → RIGHT
    img_cx_mm = float(naff.apply_affine(affine, [seg_nii.shape[0] / 2, 0, 0])[0])
    cx = wt["cx_mm"]
    if cx is not None:
        row["laterality"] = "Left" if cx < img_cx_mm else "Right"
    else:
        row["laterality"] = None

    # 부가 플래그
    row["ET_present"]  = int(et["vol_mm3"] > 0)
    row["NCR_present"] = int(ncr["vol_mm3"] > 0)
    # WT bbox 종횡비 (elongation)
    spans = [wt["bbox_span_x"], wt["bbox_span_y"], wt["bbox_span_z"]]
    if all(s is not None and s > 0 for s in spans):
        row["WT_elongation"] = round(max(spans) / min(spans), 2)
    else:
        row["WT_elongation"] = None

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = args.cases if args.cases else sorted(
        p.name for p in DATA_ROOT.iterdir() if p.is_dir()
    )
    print(f"Cases: {len(cases)}")

    rows = []
    for i, case_id in enumerate(cases, 1):
        if i % 100 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {case_id}")
        row = analyze_case(case_id)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "seg_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n→ {out.relative_to(PROJECT_ROOT)}  ({len(df)} rows, {len(df.columns)} cols)")
    print("\nDone.")


if __name__ == "__main__":
    main()

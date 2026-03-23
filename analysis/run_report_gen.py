#!/usr/bin/env python3
"""
seg_summary.csv + atlas/summary.csv 병합 → 케이스별 판독문 생성 (한국어 + English).

출력:
  analysis/results/reports/{case_id}_ko.txt   ← 한국어 판독문
  analysis/results/reports/{case_id}_en.txt   ← English report
  analysis/results/report_summary.csv         ← 전체 통합 CSV (report_ko, report_en 컬럼 포함)

Usage (run from fedpod-new/):
    python analysis/run_report_gen.py
    python analysis/run_report_gen.py --cases FeTS2022_00000 FeTS2022_00002
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "analysis" / "results"
REPORTS_DIR  = RESULTS_DIR / "reports"


# ── 공통 기술어 ─────────────────────────────────────────────────────────────

def size_grade(vol_cc: float, lang: str) -> str:
    if lang == "ko":
        if vol_cc >= 150: return "대형"
        if vol_cc >= 50:  return "중형"
        return "소형"
    else:
        if vol_cc >= 150: return "large"
        if vol_cc >= 50:  return "moderate"
        return "small"


def shape_desc(elon: float | None, lang: str) -> str:
    if elon is None:
        return "형태 측정 불가" if lang == "ko" else "shape not measurable"
    if lang == "ko":
        if elon >= 2.0: return "불규칙한 형태"
        if elon >= 1.5: return "타원형"
        return "구형에 가까운 형태"
    else:
        if elon >= 2.0: return "irregular shape"
        if elon >= 1.5: return "elongated / ovoid"
        return "nearly spherical"


def et_desc(et_pct: float, et_present: int, lang: str) -> str:
    if lang == "ko":
        if not et_present:   return "조영증강 성분은 관찰되지 않음"
        if et_pct >= 30:     return "조영증강 성분이 뚜렷하게 관찰됨"
        if et_pct >= 10:     return "조영증강 성분이 일부 관찰됨"
        return "조영증강 성분이 미미하게 관찰됨"
    else:
        if not et_present:   return "No enhancing tumor component identified"
        if et_pct >= 30:     return "Prominent enhancing tumor component present"
        if et_pct >= 10:     return "Enhancing tumor component present"
        return "Minimal enhancing tumor component present"


def ncr_desc(ncr_pct: float, ncr_present: int, lang: str) -> str:
    if lang == "ko":
        if not ncr_present:  return "괴사 성분 없음"
        if ncr_pct >= 20:    return "괴사 성분이 유의하게 동반됨"
        if ncr_pct >= 5:     return "괴사 성분이 일부 동반됨"
        return "괴사 성분이 미미하게 동반됨"
    else:
        if not ncr_present:  return "No necrotic core identified"
        if ncr_pct >= 20:    return "Substantial necrotic core present"
        if ncr_pct >= 5:     return "Necrotic core present"
        return "Minimal necrotic core present"


def regions_text(atl: pd.Series) -> str:
    parts = []
    for i in [1, 2, 3]:
        r = atl.get(f"top{i}_region", "")
        p = atl.get(f"top{i}_pct", 0)
        if pd.notna(r) and r and p > 0:
            parts.append(f"{r} ({p:.1f}%)")
    return ", ".join(parts) if parts else "N/A"


# ── 한국어 판독문 ────────────────────────────────────────────────────────────

def report_ko(seg: pd.Series, atl: pd.Series) -> str:
    cid      = seg.name
    lat      = seg["laterality"]
    lobe     = atl.get("top_lobe", "미상")
    lobe_pct = float(atl.get("top_lobe_pct", 0.0))

    wt_cc   = float(seg["WT_vol_cc"]);  wt_mm3  = int(seg["WT_vol_mm3"])
    max_dim = float(seg["WT_bbox_max_dim"])
    span_x, span_y, span_z = float(seg["WT_bbox_span_x"]), float(seg["WT_bbox_span_y"]), float(seg["WT_bbox_span_z"])
    elon    = seg.get("WT_elongation")

    ncr_cc, ncr_pct = float(seg["NCR_vol_cc"]), float(seg["NCR_pct"])
    ed_cc,  ed_pct  = float(seg["ED_vol_cc"]),  float(seg["ED_pct"])
    et_cc,  et_pct  = float(seg["ET_vol_cc"]),  float(seg["ET_pct"])
    tc_cc,  tc_pct  = float(seg["TC_vol_cc"]),  float(seg["TC_pct"])
    et_p, ncr_p     = int(seg["ET_present"]),   int(seg["NCR_present"])

    dist_ncr_et = seg.get("dist_NCR_ET_mm")
    dist_wt_et  = seg.get("dist_WT_ET_mm")
    cx, cy, cz  = atl.get("centroid_x_mm", "?"), atl.get("centroid_y_mm", "?"), atl.get("centroid_z_mm", "?")
    reg = regions_text(atl)

    L = []
    L += [f"[케이스 ID] {cid}", ""]
    L += ["[소견]",
          f"{lat} {lobe} 엽에 뇌종양이 확인됩니다 (로브 점유율 {lobe_pct:.1f}%). "
          f"병변은 주로 {reg} 영역에 분포합니다.", ""]
    L += ["▸ 크기 및 형태",
          f"  전체 종양(WT) 부피: {wt_cc:.1f} cc ({wt_mm3:,} mm³) — {size_grade(wt_cc, 'ko')}",
          f"  바운딩박스: {span_x:.0f} × {span_y:.0f} × {span_z:.0f} mm  (최대 직경 {max_dim:.0f} mm)",
          f"  형태: {shape_desc(elon, 'ko')}  (elongation {elon:.2f})" if elon else "  형태: 측정 불가",
          f"  MNI 중심좌표: x={cx:+} y={cy:+} z={cz:+} mm", ""]
    L += ["▸ 세부 구성",
          f"  부종 (ED) : {ed_cc:.1f} cc  ({ed_pct:.1f}% of WT)",
          f"  종양핵(TC): {tc_cc:.1f} cc  ({tc_pct:.1f}% of WT)",
          f"    조영증강종양 (ET) : {et_cc:.1f} cc  ({et_pct:.1f}% of WT)",
          f"    괴사핵      (NCR): {ncr_cc:.1f} cc  ({ncr_pct:.1f}% of WT)"]
    if pd.notna(dist_ncr_et):
        L.append(f"  NCR–ET 중심 간 거리: {dist_ncr_et:.1f} mm")
    if pd.notna(dist_wt_et):
        L.append(f"  WT–ET  중심 간 거리: {dist_wt_et:.1f} mm")
    L.append("")
    L += ["[인상]",
          f"{lat} {lobe} 엽의 뇌종양. 전체 종양 {wt_cc:.1f} cc ({size_grade(wt_cc, 'ko')}).",
          f"  {et_desc(et_pct, et_p, 'ko')}.",
          f"  {ncr_desc(ncr_pct, ncr_p, 'ko')}.",
          f"  부종(ED)이 WT의 {ed_pct:.1f}%를 차지하며, 종양핵(TC)은 {tc_pct:.1f}%입니다."]
    return "\n".join(L)


# ── English report ───────────────────────────────────────────────────────────

def report_en(seg: pd.Series, atl: pd.Series) -> str:
    cid      = seg.name
    lat      = seg["laterality"]
    lobe     = atl.get("top_lobe", "Unknown")
    lobe_pct = float(atl.get("top_lobe_pct", 0.0))

    wt_cc   = float(seg["WT_vol_cc"]);  wt_mm3  = int(seg["WT_vol_mm3"])
    max_dim = float(seg["WT_bbox_max_dim"])
    span_x, span_y, span_z = float(seg["WT_bbox_span_x"]), float(seg["WT_bbox_span_y"]), float(seg["WT_bbox_span_z"])
    elon    = seg.get("WT_elongation")

    ncr_cc, ncr_pct = float(seg["NCR_vol_cc"]), float(seg["NCR_pct"])
    ed_cc,  ed_pct  = float(seg["ED_vol_cc"]),  float(seg["ED_pct"])
    et_cc,  et_pct  = float(seg["ET_vol_cc"]),  float(seg["ET_pct"])
    tc_cc,  tc_pct  = float(seg["TC_vol_cc"]),  float(seg["TC_pct"])
    et_p, ncr_p     = int(seg["ET_present"]),   int(seg["NCR_present"])

    dist_ncr_et = seg.get("dist_NCR_ET_mm")
    dist_wt_et  = seg.get("dist_WT_ET_mm")
    cx, cy, cz  = atl.get("centroid_x_mm", "?"), atl.get("centroid_y_mm", "?"), atl.get("centroid_z_mm", "?")
    reg = regions_text(atl)

    L = []
    L += [f"[Case ID] {cid}", ""]
    L += ["[FINDINGS]",
          f"A brain tumor is identified in the {lat.lower()} {lobe} lobe "
          f"(lobar coverage {lobe_pct:.1f}%). "
          f"The lesion predominantly involves {reg}.", ""]
    L += ["  Size and morphology:",
          f"    Whole Tumor (WT) volume : {wt_cc:.1f} cc ({wt_mm3:,} mm³) — {size_grade(wt_cc, 'en')}",
          f"    Bounding box            : {span_x:.0f} x {span_y:.0f} x {span_z:.0f} mm  "
          f"(max diameter {max_dim:.0f} mm)",
          f"    Shape                   : {shape_desc(elon, 'en')}  (elongation ratio {elon:.2f})" if elon
              else "    Shape                   : not measurable",
          f"    MNI centroid            : x={cx:+} y={cy:+} z={cz:+} mm", ""]
    L += ["  Tumor composition:",
          f"    Peritumoral Edema  (ED) : {ed_cc:.1f} cc  ({ed_pct:.1f}% of WT)",
          f"    Tumor Core         (TC) : {tc_cc:.1f} cc  ({tc_pct:.1f}% of WT)",
          f"      Enhancing Tumor  (ET) : {et_cc:.1f} cc  ({et_pct:.1f}% of WT)",
          f"      Necrotic Core   (NCR) : {ncr_cc:.1f} cc  ({ncr_pct:.1f}% of WT)"]
    if pd.notna(dist_ncr_et):
        L.append(f"    NCR–ET centroid distance : {dist_ncr_et:.1f} mm")
    if pd.notna(dist_wt_et):
        L.append(f"    WT–ET  centroid distance : {dist_wt_et:.1f} mm")
    L.append("")
    L += ["[IMPRESSION]",
          f"Brain tumor in the {lat.lower()} {lobe} lobe. "
          f"Total tumor volume {wt_cc:.1f} cc ({size_grade(wt_cc, 'en')}).",
          f"  {et_desc(et_pct, et_p, 'en')}.",
          f"  {ncr_desc(ncr_pct, ncr_p, 'en')}.",
          f"  Peritumoral edema (ED) accounts for {ed_pct:.1f}% of WT; "
          f"tumor core (TC) comprises {tc_pct:.1f}%."]
    return "\n".join(L)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()

    seg_df = pd.read_csv(RESULTS_DIR / "seg_summary.csv").set_index("Subject_ID")
    atl_df = pd.read_csv(RESULTS_DIR / "atlas" / "summary.csv").set_index("Subject_ID")

    cases = args.cases if args.cases else sorted(seg_df.index)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cases: {len(cases)}")
    rows = []
    for i, cid in enumerate(cases, 1):
        if i % 200 == 0 or i == 1:
            print(f"  [{i}/{len(cases)}] {cid}")

        if cid not in seg_df.index or cid not in atl_df.index:
            print(f"  [skip] {cid}")
            continue

        seg = seg_df.loc[cid]
        atl = atl_df.loc[cid]

        rko = report_ko(seg, atl)
        ren = report_en(seg, atl)

        (REPORTS_DIR / f"{cid}_ko.txt").write_text(rko, encoding="utf-8")
        (REPORTS_DIR / f"{cid}_en.txt").write_text(ren, encoding="utf-8")

        rows.append({
            "Subject_ID":     cid,
            "laterality":     seg["laterality"],
            "top_lobe":       atl.get("top_lobe"),
            "top_lobe_pct":   atl.get("top_lobe_pct"),
            "top1_region":    atl.get("top1_region"),
            "WT_vol_cc":      seg["WT_vol_cc"],
            "WT_max_dim_mm":  seg["WT_bbox_max_dim"],
            "WT_elongation":  seg.get("WT_elongation"),
            "NCR_vol_cc":     seg["NCR_vol_cc"],
            "ED_vol_cc":      seg["ED_vol_cc"],
            "ET_vol_cc":      seg["ET_vol_cc"],
            "TC_vol_cc":      seg["TC_vol_cc"],
            "NCR_pct":        seg["NCR_pct"],
            "ED_pct":         seg["ED_pct"],
            "ET_pct":         seg["ET_pct"],
            "TC_pct":         seg["TC_pct"],
            "ET_present":     seg["ET_present"],
            "NCR_present":    seg["NCR_present"],
            "dist_NCR_ET_mm": seg.get("dist_NCR_ET_mm"),
            "centroid_x_mm":  atl.get("centroid_x_mm"),
            "centroid_y_mm":  atl.get("centroid_y_mm"),
            "centroid_z_mm":  atl.get("centroid_z_mm"),
            "report_ko":      rko,
            "report_en":      ren,
        })

    df_out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "report_summary.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n→ {out_path.relative_to(PROJECT_ROOT)}  ({len(df_out)} rows)")
    print(f"→ {REPORTS_DIR.relative_to(PROJECT_ROOT)}/  ({len(rows) * 2} txt files, _ko/_en)")
    print("\nDone.")


if __name__ == "__main__":
    main()

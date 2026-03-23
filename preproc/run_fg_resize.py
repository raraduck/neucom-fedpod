#!/usr/bin/env python3
"""
Preprocessing: fets240/inst_01 → fg crop + resize 128³ → fets128/inst_01

Pipeline per case:
  1. T1 기준으로 foreground bounding box 계산 (non-zero region + margin)
  2. 모든 모달리티에 동일한 crop 적용
  3. 128×128×128 으로 resize
     - t1 / t1ce / t2 / flair : bilinear  (order=1)
     - seg / sub               : nearest   (order=0)
  4. 새 voxel 크기를 affine & header 에 반영하여 저장

Usage (run from fedpod-new/):
    python preproc/run_fg_resize.py                    # 전체 (기본 workers)
    python preproc/run_fg_resize.py --workers 16
    python preproc/run_fg_resize.py --cases FeTS2022_00000 FeTS2022_00002
    python preproc/run_fg_resize.py --dry-run          # 실제 저장 없이 통계만 출력
"""

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as nd_zoom

# ── 경로 설정 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT     = PROJECT_ROOT / "data" / "fets240" / "inst_01"
DST_ROOT     = PROJECT_ROOT / "data" / "fets128" / "inst_01"

TARGET_SIZE = 128
FG_MARGIN   = 10

# 모달리티 → (interpolation order)
# bilinear=1 for images, nearest=0 for masks
MODALITIES: dict[str, int] = {
    "t1":    1,
    "t1ce":  1,
    "t2":    1,
    "flair": 1,
    "seg":   0,
    "sub":   0,
}


# ── 핵심 함수 ─────────────────────────────────────────────────────────────

def fg_bbox(t1: np.ndarray, margin: int) -> tuple[slice, ...]:
    """T1 non-zero region의 bounding box + margin → slices 반환."""
    coords = np.argwhere(t1 > 0)
    if len(coords) == 0:
        return tuple(slice(0, s) for s in t1.shape)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1  # inclusive → exclusive

    return tuple(
        slice(max(0, int(lo[i]) - margin),
              min(t1.shape[i], int(hi[i]) + margin))
        for i in range(3)
    )


def compute_new_affine(src_affine: np.ndarray,
                       slices: tuple[slice, ...],
                       crop_shape: tuple[int, ...],
                       target: int) -> np.ndarray:
    """
    Crop + resize 후의 affine 계산.

    - origin : crop start 좌표로 이동
    - spacing: crop_shape / target 비율로 재조정
    """
    start     = np.array([s.start for s in slices], dtype=float)
    scale     = np.array([sh / target for sh in crop_shape])
    new_aff   = src_affine.copy()
    new_aff[:3, 3]  = src_affine[:3, :3] @ start + src_affine[:3, 3]
    new_aff[:3, :3] = src_affine[:3, :3] @ np.diag(scale)
    return new_aff


def resize_vol(vol: np.ndarray,
               crop_shape: tuple[int, ...],
               target: int,
               order: int) -> np.ndarray:
    """scipy zoom으로 target³ 로 resize. 결과가 정확히 target 크기임을 보장."""
    factors = tuple(target / sh for sh in crop_shape)
    out = nd_zoom(vol, factors, order=order, prefilter=False)

    # 부동소수점 오차로 target 보다 1 픽셀 벗어날 경우 clipping
    if out.shape != (target, target, target):
        out = out[:target, :target, :target]
        pad = [(0, max(0, target - s)) for s in out.shape]
        if any(p[1] > 0 for p in pad):
            out = np.pad(out, pad, mode="edge")

    return out


# ── 케이스 단위 처리 ──────────────────────────────────────────────────────

def process_case(args: tuple[str, bool]) -> dict:
    case_id, dry_run = args
    src_dir = SRC_ROOT / case_id
    dst_dir = DST_ROOT / case_id

    result = {"case_id": case_id, "status": "ok",
              "crop_shape": None, "new_spacing": None, "msg": ""}

    # T1 로드 → fg bbox
    t1_path = src_dir / f"{case_id}_t1.nii.gz"
    if not t1_path.exists():
        result.update(status="skip", msg="t1 not found")
        return result

    try:
        t1_nii     = nib.load(str(t1_path))
        src_affine = t1_nii.affine
        src_header = t1_nii.header
        t1_data    = np.asarray(t1_nii.dataobj, dtype=np.float32)

        slices     = fg_bbox(t1_data, FG_MARGIN)
        crop_shape = tuple(s.stop - s.start for s in slices)
        new_affine = compute_new_affine(src_affine, slices, crop_shape, TARGET_SIZE)
        new_spacing = np.sqrt((new_affine[:3, :3] ** 2).sum(axis=0))

        result["crop_shape"]  = crop_shape
        result["new_spacing"] = new_spacing.round(4).tolist()

        if dry_run:
            return result

        dst_dir.mkdir(parents=True, exist_ok=True)

        for mod, order in MODALITIES.items():
            src_path = src_dir / f"{case_id}_{mod}.nii.gz"
            if not src_path.exists():
                continue

            dtype = np.float32 if order == 1 else np.uint8
            vol   = np.asarray(nib.load(str(src_path)).dataobj, dtype=dtype)

            cropped = vol[slices]
            resized = resize_vol(cropped, crop_shape, TARGET_SIZE, order)
            if order == 0:
                resized = resized.astype(np.uint8)

            # header 업데이트
            new_hdr = src_header.copy()
            new_hdr.set_data_shape((TARGET_SIZE,) * 3)
            new_hdr.set_zooms(new_spacing)

            dst_path = dst_dir / f"{case_id}_{mod}.nii.gz"
            nib.save(nib.Nifti1Image(resized, new_affine, new_hdr), str(dst_path))

    except Exception as e:
        result.update(status="error", msg=str(e))

    return result


# ── 요약 출력 ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    ok     = [r for r in results if r["status"] == "ok"]
    skip   = [r for r in results if r["status"] == "skip"]
    errors = [r for r in results if r["status"] == "error"]

    if not ok:
        print("No successful cases.")
        return

    spacings = np.array([r["new_spacing"] for r in ok])
    crops    = np.array([r["crop_shape"]  for r in ok])

    print(f"\n── 처리 요약 ──────────────────────────────")
    print(f"  ok={len(ok)}, skip={len(skip)}, error={len(errors)}")
    print(f"  fg crop shape  mean : {crops.mean(axis=0).round(1)}  "
          f"min={crops.min(axis=0)}  max={crops.max(axis=0)}")
    print(f"  new spacing mm mean : {spacings.mean(axis=0).round(3)}  "
          f"min={spacings.min(axis=0).round(3)}  max={spacings.max(axis=0).round(3)}")
    if errors:
        print(f"  errors: {[e['case_id'] for e in errors[:5]]}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="fets240 → fg crop + resize 128³ → fets128 (correct header)"
    )
    parser.add_argument("--workers",  type=int, default=min(cpu_count(), 8))
    parser.add_argument("--cases",    nargs="+", default=None,
                        help="처리할 케이스 ID 목록 (기본값: SRC_ROOT 전체)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="저장 없이 fg crop 통계만 출력")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] 파일 저장 없이 통계만 출력합니다.")
    else:
        DST_ROOT.mkdir(parents=True, exist_ok=True)

    case_ids = args.cases if args.cases else sorted(os.listdir(SRC_ROOT))
    total    = len(case_ids)
    workers  = args.workers

    print(f"Source  : {SRC_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Target  : {DST_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Cases   : {total},  Workers: {workers},  Target size: {TARGET_SIZE}³")

    task_args = [(cid, args.dry_run) for cid in case_ids]
    results   = []

    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap(process_case, task_args), 1):
            results.append(res)
            if res["status"] != "ok":
                print(f"  [{i:4d}/{total}] {res['status'].upper():5s} "
                      f"{res['case_id']}  {res['msg']}")
            if i % 200 == 0 or i == total:
                done = sum(1 for r in results if r["status"] == "ok")
                print(f"  [{i:4d}/{total}] ok={done}")

    print_summary(results)
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sample: fets240 케이스 1개를 MNI152 T1 1mm Brain 그리드에 affine-resample.

수행 내용:
  1. fets240 affine 과 MNI affine 에서 리샘플 행렬 계산
       fets_vox = inv(fets_aff) @ mni_aff @ mni_vox
  2. scipy.ndimage.affine_transform 으로 MNI grid 에 보간
       - t1/t1ce/t2/flair : bilinear (order=1)
       - seg / sub        : nearest  (order=0)
  3. 결과를 MNI affine 으로 저장
       data/fets240-step1-align/{case_id}/{case_id}_{mod}.nii.gz
  4. 정합 품질 확인용 요약 출력

Usage (run from fedpod-new/):
    python preproc/sample_align_to_mni.py
    python preproc/sample_align_to_mni.py --case FeTS2022_00002
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy.ndimage import affine_transform

# ── 경로 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT     = PROJECT_ROOT / "data" / "fets240" / "inst_01"
MNI_PATH     = PROJECT_ROOT / "data" / "atlases" / "MNI152_T1_1mm_Brain.nii.gz"
DST_ROOT     = PROJECT_ROOT / "data" / "fets240-step1-align"

MODALITIES: dict[str, int] = {
    "t1":    1,
    "t1ce":  1,
    "t2":    1,
    "flair": 1,
    "seg":   0,
    "sub":   0,
}


# ── 핵심 함수 ─────────────────────────────────────────────────────────────

def build_resample_params(src_affine: np.ndarray,
                          ref_affine: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    scipy.ndimage.affine_transform 에 넘길 (matrix, offset) 계산.

    변환 관계:
        src_vox = inv(src_aff) @ ref_aff @ ref_vox
    => matrix = inv(src_aff)[:3,:3] @ ref_aff[:3,:3]
       offset = (inv(src_aff) @ ref_aff)[:3, 3]
    """
    M    = np.linalg.inv(src_affine) @ ref_affine   # 4×4
    mat  = M[:3, :3]
    off  = M[:3,  3]
    return mat, off


def resample_to_ref(vol: np.ndarray,
                    src_affine: np.ndarray,
                    ref_shape: tuple[int, ...],
                    ref_affine: np.ndarray,
                    order: int,
                    cval: float = 0.0) -> np.ndarray:
    """src 볼륨을 ref grid 에 리샘플."""
    mat, off = build_resample_params(src_affine, ref_affine)
    out = affine_transform(
        vol,
        matrix=mat,
        offset=off,
        output_shape=ref_shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return out


# ── 케이스 처리 ───────────────────────────────────────────────────────────

def align_case(case_id: str, mni_nii: nib.Nifti1Image) -> None:
    src_dir = SRC_ROOT / case_id
    dst_dir = DST_ROOT / case_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    ref_affine = mni_nii.affine
    ref_shape  = mni_nii.shape[:3]   # (182, 218, 182)

    # T1 로 affine 읽기
    t1_path = src_dir / f"{case_id}_t1.nii.gz"
    if not t1_path.exists():
        print(f"[SKIP] {case_id}: t1 not found")
        return

    src_nii    = nib.load(str(t1_path))
    src_affine = src_nii.affine

    # ── 정합 전 정보 ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Case : {case_id}")
    print(f"  src  shape  : {src_nii.shape}")
    print(f"  src  zooms  : {np.array(src_nii.header.get_zooms()).round(3)}")
    print(f"  src  orient : {aff2axcodes(src_affine)}")
    print(f"  src  origin : {src_affine[:3, 3].round(2)}  (mm, vox[0,0,0])")
    print(f"  ref  shape  : {ref_shape}")
    print(f"  ref  orient : {aff2axcodes(ref_affine)}")
    print(f"  ref  origin : {ref_affine[:3, 3].round(2)}  (mm, vox[0,0,0])")

    mat, off = build_resample_params(src_affine, ref_affine)
    print(f"\n  resample matrix:\n{np.round(mat, 4)}")
    print(f"  resample offset: {off.round(4)}")

    # ── 각 모달리티 리샘플 & 저장 ──────────────────────────────────────────
    new_hdr = mni_nii.header.copy()

    for mod, order in MODALITIES.items():
        src_path = src_dir / f"{case_id}_{mod}.nii.gz"
        if not src_path.exists():
            print(f"  [skip] {mod} not found")
            continue

        dtype = np.float32 if order == 1 else np.uint8
        vol   = np.asarray(nib.load(str(src_path)).dataobj, dtype=dtype)

        resampled = resample_to_ref(vol, src_affine, ref_shape, ref_affine, order)
        if order == 0:
            resampled = resampled.astype(np.uint8)

        new_hdr.set_data_shape(ref_shape)
        dst_path = dst_dir / f"{case_id}_{mod}.nii.gz"
        nib.save(nib.Nifti1Image(resampled, ref_affine, new_hdr), str(dst_path))
        print(f"  [{mod:5s}] saved → {dst_path.relative_to(PROJECT_ROOT)}"
              f"  nonzero={int((resampled>0).sum())}")

    # ── 정합 후 검증 ───────────────────────────────────────────────────────
    _verify(case_id, dst_dir, mni_nii)


def _verify(case_id: str, dst_dir: Path, mni_nii: nib.Nifti1Image) -> None:
    t1_dst = dst_dir / f"{case_id}_t1.nii.gz"
    if not t1_dst.exists():
        return

    out   = nib.load(str(t1_dst))
    t1    = np.asarray(out.dataobj, dtype=np.float32)
    mni_t = np.asarray(mni_nii.dataobj, dtype=np.float32)

    brain_mask = mni_t > 0
    overlap    = (t1 > 0) & brain_mask
    coverage   = overlap.sum() / brain_mask.sum() * 100

    print(f"\n  [검증]")
    print(f"  출력 shape  : {out.shape}  (MNI grid ✓)" if out.shape == mni_nii.shape
          else f"  출력 shape  : {out.shape}  ← MISMATCH!")
    print(f"  출력 affine :\n{out.affine}")
    print(f"  MNI brain voxel 중 T1 비율: {coverage:.1f}%")
    print(f"  (100% 에 가까울수록 정합 양호)")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="fets240 sample → MNI152 grid align (affine resample)"
    )
    parser.add_argument(
        "--case", default=None,
        help="처리할 케이스 ID (기본값: SRC_ROOT 첫 번째 케이스)"
    )
    args = parser.parse_args()

    mni_nii = nib.load(str(MNI_PATH))
    print(f"MNI ref : {MNI_PATH.relative_to(PROJECT_ROOT)}")
    print(f"         shape={mni_nii.shape}  "
          f"orient={aff2axcodes(mni_nii.affine)}  "
          f"zooms={mni_nii.header.get_zooms()}")

    if args.case:
        case_id = args.case
    else:
        cases = sorted(p.name for p in SRC_ROOT.iterdir() if p.is_dir())
        case_id = cases[0]
        print(f"\n(--case 미지정 → 첫 케이스 사용: {case_id})")

    align_case(case_id, mni_nii)
    print("\nDone.")


if __name__ == "__main__":
    main()

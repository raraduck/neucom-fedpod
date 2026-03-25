# neucom-fedpod

뇌종양 MRI 데이터(FeTS2022)를 대상으로 한 전처리·분석·판독문 생성 파이프라인.

---

## 목차

1. [데이터 구조](#1-데이터-구조)
2. [전처리 파이프라인](#2-전처리-파이프라인)
3. [MNI 정합 (Kubernetes Jobs)](#3-mni-정합-kubernetes-jobs)
4. [분석 스크립트](#4-분석-스크립트)
5. [결과물 요약](#5-결과물-요약)
6. [디렉토리 구조](#6-디렉토리-구조)

---

## 1. 데이터 구조

### 원본 데이터 (`data/fets240/inst_01/`)

| 항목 | 내용 |
|------|------|
| 케이스 수 | 1,251개 |
| 형식 | NIfTI 1mm isotropic, 240×240×155 voxel |
| 모달리티 | `_t1`, `_t1ce`, `_t2`, `_flair` |
| 세그멘테이션 | `_seg` (WT binary), `_sub` (NCR/ED/ET 다중 라벨) |

**세그멘테이션 라벨 정의**

| 파일 | 라벨 | 의미 |
|------|------|------|
| `_seg.nii.gz` | 1 | Whole Tumor (WT) |
| `_sub.nii.gz` | 1 | Necrotic Core (NCR) |
| `_sub.nii.gz` | 2 | Peritumoral Edema (ED) |
| `_sub.nii.gz` | 4 | Enhancing Tumor (ET) |

### 전처리 결과 데이터

| 폴더 | 내용 |
|------|------|
| `data/fets128/inst_01/` | fets240 → 128³ crop+resize (foreground 기반) |
| `data/fets128-mni-reg/` | fets128 → MNI152 FLIRT 정합 결과 (1,251케이스) |
| `data/fets128-mni-reg-failed/` | 축 반전 실패 79케이스 원본 백업 |
| `data/atlases/` | MNI152_T1_1mm_Brain.nii.gz + Harvard-Oxford atlas |

---

## 2. 전처리 파이프라인

### `preproc/run_fg_resize.py` — fets240 → fets128

- foreground bounding box 기준으로 crop → 128×128×128 resize
- 모든 모달리티 및 마스크 동일 변환 적용

### `analysis/run_seg_stats.py` — 세그멘테이션 통계

- WT/NCR/ED/ET voxel 수, 물리 볼륨(mm³), centroid, bbox 추출
- 결과: `analysis/results/{partition}/seg_stats.csv`

### `analysis/compare_seg_stats.py` — fets240 vs fets128 비교

- 전처리 전후 부피·centroid 보존 여부 검증
- 결과: `analysis/results/compare/{partition}/compare_stats.csv`

---

## 3. MNI 정합 (Kubernetes Jobs)

모든 Job은 `nodeName: gn144` 고정 (RWO PVC `fedpod` 마운트 노드).
이미지: `192.168.0.80:30002/dwnkim/myfsl:v1.0` (FSL 5.0 포함)

### 실행 순서

```
1. k8s/flirt-all-job.yaml        # 전체 1,251케이스 FLIRT 정합
2. k8s/backup-failed-job.yaml    # 실패 79케이스 백업
3. k8s/flirt-retry-v1-job.yaml   # 79케이스 재처리 (fslreorient2std 적용)
4. k8s/replace-v1-job.yaml       # fets128-mni-reg/ 에 v1 결과 반영
```

### Job 목록

| 파일 | 역할 | completions / parallelism |
|------|------|--------------------------|
| `flirt-all-job.yaml` | 전체 1,251케이스 FLIRT (12-DOF, mutualinfo) | 1251 / 16 |
| `flirt-sample-job.yaml` | 샘플 3케이스 테스트 | 3 / 3 |
| `flirt-retry-sample-job.yaml` | 재처리 방법 샘플 검증 | 3 / 3 |
| `flirt-retry-v1-job.yaml` | 실패 79케이스 재처리 **(fslreorient2std)** | 79 / 16 |
| `flirt-retry-job.yaml` | 대안 재처리 (swapdim, 미사용) | 79 / 16 |
| `backup-failed-job.yaml` | 실패 케이스 백업 → `fets128-mni-reg-bak/` | 1 / 1 |
| `replace-v1-job.yaml` | v1 결과로 교체 + bak → failed 이름 변경 | 1 / 1 |
| `copy-atlas-job.yaml` | FSL 컨테이너에서 atlas NIfTI/XML 추출 | 1 / 1 |
| `fix-atlas-perm-job.yaml` | atlas 폴더 소유권 수정 (root → jovyan) | 1 / 1 |

### FLIRT 실패 원인 및 수정

**원인**: fets128 데이터가 LPS (det>0, NEUROLOGICAL) 방향이고 MNI는 LAS (det<0, RADIOLOGICAL) 방향.
축 방향 차이로 FLIRT가 국소 최솟값(180° 회전)으로 수렴하는 케이스 발생.

**탐지**: 변환 행렬 대각원소 `M[0,0] > 0` 또는 `scale_ratio > 1.5` → 79케이스 flagged
(`analysis/results/atlas/flirt_qc.csv`)

**수정**: `fslreorient2std` 를 FLIRT 이전에 적용해 표준 방향으로 재정렬 후 정합 → 79/79 성공

### FLIRT 파라미터

```bash
flirt -dof 12 -cost mutualinfo -bins 256 -interp trilinear
# 마스크: -interp nearestneighbour
```

---

## 4. 분석 스크립트

### `analysis/run_seg_summary.py` — 원본 물리 수치 추출

fets240 원본 공간에서 WT·NCR·ED·ET의 물리적 측정값 추출.

```bash
python analysis/run_seg_summary.py                          # 전체 1,251케이스
python analysis/run_seg_summary.py --cases FeTS2022_00000   # 특정 케이스
```

**출력** (`analysis/results/seg_summary.csv`, 1,251행 × 48열)

| 컬럼 그룹 | 주요 컬럼 |
|-----------|----------|
| 볼륨 | `WT/NCR/ED/ET/TC_vol_mm3`, `_vol_cc` |
| 중심좌표 | `WT/NCR/ED/ET_cx/cy/cz_mm` (native 공간) |
| 바운딩박스 | `WT_bbox_span_x/y/z`, `WT_bbox_max_dim`, `WT_elongation` |
| 비율 | `NCR/ED/ET/TC_pct` (WT 대비 %) |
| 거리 | `dist_NCR_ET_mm`, `dist_WT_NCR_mm`, `dist_WT_ET_mm` |
| 플래그 | `laterality`, `ET_present`, `NCR_present` |

### `analysis/run_atlas_mapping.py` — Harvard-Oxford 아틀라스 매핑

MNI 정합 결과에 대해 Harvard-Oxford Cortical/Subcortical atlas 매핑.

```bash
python analysis/run_atlas_mapping.py
python analysis/run_atlas_mapping.py --cases FeTS2022_00000
```

**출력**
- `analysis/results/atlas/summary.csv` (1,251행): 로브·laterality·top-3 region
- `analysis/results/atlas/{case_id}_atlas.csv`: 케이스별 region overlap 상세

**laterality 기준**: MNI affine `x = -1×voxel + 90` → `x_mm > 0 = Right`, `x_mm < 0 = Left`

### `analysis/run_report_gen.py` — 판독문 생성

`seg_summary.csv` + `atlas/summary.csv` 병합 → 한국어·English 판독문 자동 생성.

```bash
python analysis/run_report_gen.py
python analysis/run_report_gen.py --cases FeTS2022_00000
```

**출력**
- `analysis/results/reports/{case_id}_ko.txt`: 한국어 판독문
- `analysis/results/reports/{case_id}_en.txt`: English report
- `analysis/results/report_summary.csv`: 통합 CSV (`report_ko`, `report_en` 컬럼 포함)

---

## 5. 결과물 요약

### 데이터셋 통계 (1,251케이스)

**볼륨 분포 (원본 fets240 기준)**

| 분위 | WT (cc) |
|------|---------|
| P25  | 48.3    |
| P50  | 89.3    |
| P75  | 136.7   |
| Mean | 108.6   |

**구성 비율 (평균)**

| 성분 | 평균 비율 |
|------|----------|
| Peritumoral Edema (ED) | 62.6% |
| Enhancing Tumor (ET)   | 23.7% |
| Necrotic Core (NCR)    | 13.6% |

**로브 분포 (MNI-registered)**

| Lobe     | 케이스 수 | 비율  |
|----------|----------|-------|
| Frontal  | 424      | 33.9% |
| Temporal | 411      | 32.9% |
| Parietal | 269      | 21.5% |
| Occipital| 118      |  9.4% |
| Other    |  26      |  2.1% |

**Laterality**: Left 658 (52.6%) / Right 593 (47.4%)
**ET 없음 (ET_present=0)**: 33케이스 (2.6%)

### 판독문 예시

```
[Case ID] FeTS2022_00000

[FINDINGS]
A brain tumor is identified in the left Frontal lobe (lobar coverage 41.7%).
The lesion predominantly involves Left Cerebral White Matter (43.7%),
Left Cerebral Cortex (39.6%), Frontal Orbital Cortex (13.6%).

  Size and morphology:
    Whole Tumor (WT) volume : 57.3 cc (57,305 mm³) — moderate
    Bounding box            : 45 x 75 x 46 mm  (max diameter 75 mm)
    Shape                   : elongated / ovoid  (elongation ratio 1.67)
    MNI centroid            : x=-16.7 y=+28.5 z=-5.4 mm

  Tumor composition:
    Peritumoral Edema  (ED) : 12.8 cc  (22.4% of WT)
    Tumor Core         (TC) : 44.5 cc  (77.6% of WT)
      Enhancing Tumor  (ET) : 32.7 cc  (57.1% of WT)
      Necrotic Core   (NCR) : 11.7 cc  (20.5% of WT)

[IMPRESSION]
Brain tumor in the left Frontal lobe. Total tumor volume 57.3 cc (moderate).
  Prominent enhancing tumor component present.
  Substantial necrotic core present.
  Peritumoral edema (ED) accounts for 22.4% of WT; tumor core (TC) comprises 77.6%.
```

---

## 6. 디렉토리 구조

```
fedpod-new/
├── CLAUDE.md                        ← 프로젝트 가이드 (FL 학습 설정 등)
├── README.md                        ← 이 파일
├── .gitignore
│
├── k8s/                             ← Kubernetes Job 매니페스트
│   ├── flirt-all-job.yaml           ← 전체 1,251케이스 MNI 정합
│   ├── flirt-retry-v1-job.yaml      ← 실패 79케이스 재처리 (fslreorient2std)
│   ├── backup-failed-job.yaml       ← 실패 케이스 백업
│   ├── replace-v1-job.yaml          ← v1 결과 교체
│   ├── copy-atlas-job.yaml          ← FSL atlas 파일 추출
│   ├── fix-atlas-perm-job.yaml      ← atlas 폴더 권한 수정
│   ├── flirt-sample-job.yaml        ← 샘플 테스트
│   ├── flirt-retry-sample-job.yaml  ← 재처리 샘플 테스트
│   └── flirt-retry-job.yaml         ← 대안 재처리 (미사용)
│
├── preproc/
│   └── run_fg_resize.py             ← fets240 → fets128 crop+resize
│
├── analysis/
│   ├── run_seg_stats.py             ← seg 통계 추출
│   ├── compare_seg_stats.py         ← fets240 vs fets128 비교
│   ├── run_atlas_mapping.py         ← Harvard-Oxford atlas 매핑
│   ├── run_seg_summary.py           ← 원본 물리 수치 추출 (볼륨·bbox·거리)
│   ├── run_report_gen.py            ← 한국어/English 판독문 생성
│   └── results/
│       ├── seg_summary.csv          ← 1,251케이스 물리 수치 (48컬럼)
│       ├── report_summary.csv       ← 1,251케이스 통합 판독문 CSV
│       ├── atlas/
│       │   ├── summary.csv          ← 로브·laterality·top region
│       │   ├── flirt_qc.csv         ← FLIRT QC (79케이스 실패 탐지)
│       │   └── {case_id}_atlas.csv  ← 케이스별 region overlap
│       └── reports/
│           ├── {case_id}_ko.txt     ← 한국어 판독문 (1,251개)
│           └── {case_id}_en.txt     ← English report (1,251개)
│
├── experiments/
│   ├── partition1/fets_split.csv
│   └── partition2/fets_split.csv
│
├── scripts/                         ← FL 학습 코드 (준비 중)
│   ├── dsets/
│   ├── models/
│   └── utils/
│
└── data/                            ← gitignore (무거운 데이터)
    ├── atlases/                     ← MNI152 + Harvard-Oxford (tracked)
    ├── fets240/inst_01/             ← 원본 1,251케이스
    ├── fets128/inst_01/             ← 전처리 결과
    ├── fets128-mni-reg/             ← MNI 정합 결과
    └── fets128-mni-reg-failed/      ← 실패 79케이스 백업
```

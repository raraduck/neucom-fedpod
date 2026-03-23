# FedPOD — Federated Learning for Medical Image Segmentation

## 프로젝트 목표

`fedpod-old`의 스크립트 구조를 분석하여 `fedpod-new`에 **깔끔하고 가볍게** 재작성한다.
진입점은 `run_train.sh`이며, `data/fets` 하위의 뇌종양 MRI 데이터를 학습한다.

---

## 디렉토리 구조 (목표)

```
fedpod-new/
├── CLAUDE.md                  ← 이 파일
├── run_train.sh               ← 학습 진입점 (shell)
├── run_aggregation.sh         ← 서버 집계 진입점 (shell)
├── data/
│   └── fets/
│       ├── inst_00/           ← 빈 디렉토리 (예약)
│       └── inst_01/           ← 기관 01, 1251개 케이스
│           ├── FeTS2022_00206/
│           │   ├── FeTS2022_00206_t1.nii.gz
│           │   ├── FeTS2022_00206_t1ce.nii.gz
│           │   ├── FeTS2022_00206_t2.nii.gz
│           │   ├── FeTS2022_00206_flair.nii.gz
│           │   ├── FeTS2022_00206_sub.nii.gz   ← 학습 타겟 (subcortical)
│           │   └── FeTS2022_00206_seg.nii.gz   ← 원본 세그멘테이션
│           └── ...
└── scripts/
    ├── run_train.py           ← python 진입점
    ├── run_aggregation.py     ← 집계 진입점
    ├── app.py                 ← 핵심 App 클래스 (train/infer/forward)
    ├── aggregator.py          ← fedavg / fedpod 등 집계 알고리즘
    ├── models/
    │   └── unet3d.py          ← 3D U-Net 구현
    ├── dsets/
    │   └── fets.py            ← FETS 데이터셋 로더
    └── utils/
        ├── args.py            ← argparse 설정
        ├── loss.py            ← Dice+BCE 손실 함수
        ├── metrics.py         ← Dice, HD95 등 평가 지표
        ├── misc.py            ← NIfTI I/O, subject 목록 관리
        └── optim.py           ← optimizer / scheduler 선택
```

---

## 데이터 구조 (`data/fets`)

- **기관(institution)**: `inst_01/` — 현재 1251개 케이스 보유
- **케이스 ID 형식**: `FeTS2022_XXXXX`
- **입력 모달리티** (4채널 기본):
  - `_t1.nii.gz` — T1 MRI
  - `_t1ce.nii.gz` — T1 contrast-enhanced
  - `_t2.nii.gz` — T2 MRI
  - `_flair.nii.gz` — FLAIR
- **학습 타겟**:
  - `_sub.nii.gz` — subcortical 세그멘테이션 마스크 (기본 학습 타겟)
  - `_seg.nii.gz` — 원본 종양 세그멘테이션 (참고용)
- **형식**: NIfTI compressed (`.nii.gz`), 3D 볼륨

### 빠른 테스트용 케이스 예시
```
inst_01/FeTS2022_00206/   inst_01/FeTS2022_01426/
inst_01/FeTS2022_01328/   inst_01/FeTS2022_01202/
```

---

## 학습 실행 방법

### 기본 실행 (단일 기관, 단일 라운드)

```bash
bash run_train.sh \
  -S 42 \
  -s 0 \
  -f 5 \
  -m "[20]" \
  -g 1 \
  -Z 1 \
  -L 1 \
  -J test_job \
  -R 1 \
  -r 0 \
  -E 30 \
  -e 0 \
  -i 1 \
  -c experiments/fets_split.csv \
  -M none \
  -p 1.0 \
  -D data/fets \
  -d fets \
  -C "[t1,t1ce,t2,flair]" \
  -G "[[1,2,3]]" \
  -N "[sub]" \
  -I 0 \
  -a fedavg
```

### 파라미터 설명

| 옵션 | 설명 | 예시 |
|------|------|------|
| `-S` | 랜덤 시드 | `42` |
| `-s` | 추론 결과 저장 (0/1) | `0` |
| `-f` | 검증 주기 (epoch 단위) | `5` |
| `-m` | LR decay 에폭 리스트 | `"[20]"` |
| `-g` | GPU 사용 (0/1) | `1` |
| `-Z` | Zoom 증강 (0/1) | `1` |
| `-L` | 좌우 반전 증강 (0/1) | `1` |
| `-J` | 작업 이름 (체크포인트 저장 경로에 사용) | `test_job` |
| `-R` | 전체 라운드 수 | `3` |
| `-r` | 현재 라운드 번호 (0-indexed) | `0` |
| `-E` | 전체 에폭 수 | `30` |
| `-e` | 현재 에폭 오프셋 | `0` |
| `-i` | 기관 ID | `1` |
| `-c` | 데이터 분할 CSV 경로 | `experiments/fets_split.csv` |
| `-M` | 초기 모델 가중치 경로 (`none` = 신규 초기화) | `none` |
| `-p` | 데이터 사용 비율 (0.0~1.0) | `1.0` |
| `-D` | 데이터 루트 경로 | `data/fets` |
| `-d` | 데이터셋 이름 | `fets` |
| `-C` | 입력 채널 이름 목록 | `"[t1,t1ce,t2,flair]"` |
| `-G` | 레이블 그룹 (2D int 리스트) | `"[[1,2,3]]"` |
| `-N` | 레이블 이름 목록 | `"[sub]"` |
| `-I` | 레이블 인덱스 | `0` |
| `-a` | FL 알고리즘 | `fedavg` |
| `-u` | FedProx mu 값 | `0.001` |

---

## 데이터 분할 CSV 형식

`experiments/` 폴더에 위치. `run_train.sh`의 `-c` 옵션으로 지정.

```csv
Partition_ID,Subject_ID,TrainOrVal,Measurement_ID,R0,R1,...,R50
1,FeTS2022_01341,train,0,,,,...
1,FeTS2022_01333,val,0,,,,...
```

| 컬럼 | 설명 |
|------|------|
| `Partition_ID` | 기관 번호 (inst_01 → 1) |
| `Subject_ID` | 케이스 ID |
| `TrainOrVal` | `train` / `val` / `test` |
| `Measurement_ID` | 사용 안 함 (0) |
| `R0~R50` | 라운드별 메트릭 저장 공간 |

---

## 모델 아키텍처

- **모델**: 3D U-Net (ResidualBlock 기반)
- **채널**: `[32, 64, 128, 256]`
- **입력 크기**: `[128, 128, 128]` (zoom crop 후)
- **배치 크기**: 1 (의료 영상 특성상)
- **정규화**: InstanceNorm3D
- **Deep Supervision**: 1 레이어
- **출력**: 다채널 sigmoid (채널 수 = 레이블 그룹 수)

---

## 지원 FL 알고리즘

| 알고리즘 | 설명 |
|----------|------|
| `fedavg` | Federated Averaging (가중 평균) |
| `fedprox` | FedProx (proximal term 추가, `-u` 로 mu 조절) |
| `fedpod` | FedPOD (Over-the-air Decomposition 기반 집계) |
| `fedwavg` | Weighted FedAVG 변형 |
| `fedpid` | Primal-Dual 방법 |

---

## 체크포인트 구조

```
states/{job_name}/R{rounds:02}r{round:02}/models/
  R{round:02}r{round:02}_last.pth    ← 마지막 에폭 모델
  R{round:02}r{round:02}_best.pth    ← 최적 검증 모델
```

각 체크포인트에 저장되는 내용:
- `model`: 모델 state_dict
- `args`: 학습 설정 전체
- `P`: 학습 샘플 수 (집계 가중치에 사용)
- `I`: 학습 전 손실 (pre-training loss)
- `D`: 개선량 `I - post_loss` (FedPOD 집계에 사용)
- `pre_metrics` / `post_metrics`: 검증 지표
- `time`: 학습 소요 시간

---

## 재작성 원칙 (fedpod-new 코딩 가이드)

1. **단순하게**: 불필요한 인자, 미사용 코드 경로 제거
2. **파일 분리 명확하게**: 하나의 파일은 하나의 역할
3. **레거시 데이터셋 제외**: CC359, PPMI 등 제외, FETS만 지원
4. **중복 스크립트 금지**: `run_train1.sh`, `run_train2.sh`, `*_bak.sh` 등 없이 `run_train.sh` 하나만
5. **인자 파서 통합**: `utils/args.py` 하나로 관리
6. **의존성 최소화**: `torch`, `nibabel`, `monai`, `numpy`, `pandas` 만 사용

---

## 주요 의존성

```
torch >= 2.0
nibabel
monai
numpy
pandas
```

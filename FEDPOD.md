# FedPOD — 파이프라인 개요

## 전체 흐름

```
[Stage 1] WT 탐지 (4ch → 1 class)
    run_init.sh   → init.pth 생성
    run_stage1.sh → 33개 기관 로컬 학습 + FedAVG 집계
         ↓
[Committee] 불확실성 평가 → global_priority.json
    run_committee_stage1.sh → run_committee_global.sh
         ↓
[Stage 2] 하위 영역 분할 (5ch → 3 class: NCR/ED/ET)
    run_init.sh         → init.pth 생성
    run_stage2_compare.sh → AL vs RND, 6 조건 병행 실험
```

전체 파이프라인은 `run_e2e.sh` 로 일괄 실행하거나
각 단계를 개별 sh 스크립트로 실행합니다.

---

## 스크립트 계층

```
run_e2e.sh
 ├─ run_init.sh           → scripts/run_init.py
 ├─ run_stage1.sh
 │    ├─ run_train.sh (×33)   → scripts/run_train.py
 │    └─ run_aggregation.sh   → scripts/run_aggregation.py
 ├─ run_committee_stage1.sh
 │    └─ run_committee_global.sh → scripts/run_committee_global.py
 └─ run_stage2_compare.sh
      ├─ run_init.sh
      ├─ run_train.sh (×3 inst × 6 cond × N rounds)
      └─ run_aggregation.sh (×6 cond × N rounds)
```

개별 기관 committee 평가는 `run_committee.sh → scripts/run_committee.py` 사용.

---

## 핵심 파라미터

### 공통

| 파라미터 | 옵션 | 설명 |
|---------|------|------|
| `JOB` | `-J` | 실험 이름 (체크포인트 경로에 사용) |
| `ROUNDS` | `-R` | FL 전체 라운드 수 |
| `ROUND` | `-r` | 현재 라운드 (0-indexed) |
| `EPOCHS` | `-E` | 에폭 수 (총 또는 라운드 끝 에폭) |
| `EPOCH` | `-e` | 에폭 오프셋 (cosine 스케줄 연속성) |
| `MODEL` | `-M` | 초기 모델 경로 (`none` = 신규 초기화) |
| `INST` | `-i` | 기관 ID |
| `ALGO` | `-a` | FL 알고리즘 (`fedavg`/`fedprox`/`fedpod`) |

### 데이터

| 파라미터 | 옵션 | Stage 1 기본값 | Stage 2 기본값 |
|---------|------|---------------|---------------|
| `CHAN` | `-C` | `[t1,t1ce,t2,flair]` | `[t1,t1ce,t2,flair,seg]` |
| `LGRP` | `-G` | `[[1,2,4]]` | `[[1],[2],[4]]` |
| `LNAM` | `-N` | `[wt]` | `[ncr,ed,et]` |
| `LIDX` | `-I` | `[1]` | `[1,2,4]` |
| `MASK` | `-X` | `[]` | `[seg]` |

### 학습 하이퍼파라미터

| 파라미터 | 옵션 | 기본값 |
|---------|------|--------|
| `LR` | `-Q` | `5e-3` |
| `BATCH` | `-B` | `2` |
| `SCHEDULER` | `-v` | `cosine` |
| `NPROC` | `-W` | `1` (DDP 시 GPU 수) |

---

## 출력물 경로

```
states/{JOB}/
  R{R:02}r{r:02}/models/
    R{R:02}r{r:02}_last.pth   ← 마지막 에폭 체크포인트
    R{R:02}r{r:02}_best.pth   ← 최적 검증 모델

states/{EXP}/s1/global/
  init.pth                    ← Stage 1 공유 초기 모델
  R01r01/models/R01r01_agg.pth ← Stage 1 집계 모델
  global_priority.json         ← Committee 불확실성 점수

states/{EXP}/s2/global/
  init.pth                    ← Stage 2 공유 초기 모델
  R{R}r{r}/models/R{R}r{r}_agg.pth

results/
  {EXP}_stage2_compare.txt    ← AL vs RND 비교 요약
```

---

## 체크포인트 내용

각 `.pth` 파일에 저장되는 키:

| 키 | 내용 |
|----|------|
| `model` | 모델 state_dict |
| `args` | 학습 설정 전체 (argparse Namespace) |
| `P` | 학습 샘플 수 (집계 가중치 계산용) |
| `I` | pre-training loss |
| `D` | 개선량 = I - post_loss (FedPOD 집계용) |
| `pre_metrics` | 학습 전 검증 지표 |
| `post_metrics` | 학습 후 검증 지표 (`dice`, `dice_per_class`) |
| `time` | 학습 소요 시간(초) |

---

## FL 알고리즘

| 알고리즘 | 설명 | 특이사항 |
|----------|------|---------|
| `fedavg` | 샘플 수 가중 평균 | 기본값 |
| `fedprox` | proximal term 추가 | `-u`(mu) 파라미터 |
| `fedpod` | Over-the-Air Decomposition | `D`(개선량) 기반 가중치 |
| `fedwavg` | Weighted FedAVG 변형 | |
| `fedpid` | Primal-Dual | |

---

## 로컬 단일 실행 예시

```bash
# Stage 1 전체 실험 (단일 노드)
bash run_init.sh -A p2_formal/s1/global -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]"
bash run_stage1.sh -E 60 -W 4 -f 10 -a fedwavg

# Committee
bash run_committee_stage1.sh -k texture_bald

# Stage 2 전체 실험
bash run_init.sh -A p2_formal/s2/global \
  -C "[t1,t1ce,t2,flair,seg]" -G "[[1,2,4],[1,4],[4]]"
bash run_stage2_compare.sh -E 3 -R 20 -W 4 -f 1

# 또는 E2E 일괄
bash run_e2e.sh -E 60 -R 20 -W 4 -f 10 -a fedwavg
```

---

## 의존성

```
torch >= 2.0        # DDP, sliding_window_inference
monai >= 1.3        # transforms, sliding_window_inference
nibabel >= 5.0      # NIfTI I/O
numpy
pandas              # split CSV
scipy >= 1.10       # monai RandRotated (scipy.ndimage)
natsort             # 파일 정렬
```

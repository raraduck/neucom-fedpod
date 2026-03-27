# FedPOD — Federated Active Learning (FAL) 파이프라인

## 개념

FedPOD FAL은 **2단계 연합 학습** 파이프라인입니다.

1. **Stage 1**: 모든 기관이 Whole Tumor(WT) 탐지 모델을 공동 학습
2. **Committee**: Stage 1 모델들이 전체 학습 데이터에 대해 불확실성을 평가 → 각 케이스의 우선순위(priority) 생성
3. **Stage 2**: 우선순위에 따라 케이스를 선별(Active Learning)하여 하위 영역(NCR/ED/ET) 세그멘테이션 학습

**핵심 가설**: 불확실성이 높은 케이스를 우선 선별하면, 동일 라벨링 비용(10~30%)으로 랜덤 선별 대비 높은 성능을 달성할 수 있다.

---

## 단계별 상세

### Stage 1 — WT 탐지

- **입력**: 4채널 MRI (`t1, t1ce, t2, flair`)
- **타겟**: `sub.nii.gz` → label 1+2+4 합집합 (binary WT mask)
- **라운드**: 1회 (여러 에폭)
- **기관 수**: 33개 (partition2 전체)
- **스크립트**: `run_stage1.sh`

```bash
bash run_init.sh -A p2_formal/s1/global \
  -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]"
bash run_stage1.sh -E 60 -W 4 -a fedwavg
```

출력:
- `states/p2_formal/s1/inst{N}/R01r00/models/R01r00_last.pth`
- `states/p2_formal/s1/global/R01r01/models/R01r01_agg.pth`

---

### Committee — 불확실성 평가

각 기관의 Stage 1 모델(K개)이 **전체 학습 케이스**를 평가합니다.
K개 확률 맵의 불일치(disagreement)를 이용해 케이스별 불확실성 점수를 산출합니다.

#### score_key 옵션

| 키 | 설명 |
|----|------|
| `bald_mi` | BALD Mutual Information (인식론적 불확실성) |
| `roi_bald` | ROI 내부 BALD (종양 영역만) |
| `variance` | 모델 간 예측 분산 |
| `mean_dice` | 위원회 평균 Dice (GT 필요) |
| `texture_bald` | BALD + texture 다양성 가중합 **(기본값, 권장)** |

#### texture_bald

```
score = bald_mi + λ × texture_diversity
```

- `bald_mi`: 인식론적 불확실성 (모델 간 불일치)
- `texture_diversity`: 케이스의 texture 특징이 기존 선택 케이스들과 얼마나 다른가
- `λ` (`-L`): 다양성 가중치 (기본 0.5)

```bash
bash run_committee_stage1.sh -k texture_bald -L 0.5
```

출력: `states/p2_formal/s1/global/global_priority.json`

```json
{
  "cases":   ["FeTS2022_00206", "FeTS2022_01328", ...],
  "scores":  [0.87, 0.63, ...],
  "weights": [1.92, 1.37, ...]
}
```

---

### Stage 2 — 하위 영역 분할 (AL vs RND 비교)

- **입력**: 5채널 MRI (`t1, t1ce, t2, flair, seg`)
  - `seg`: Stage 1 WT 예측을 binary mask로 사용 (annotation proxy)
- **타겟**: `sub.nii.gz` → label 1(NCR), 2(ED), 4(ET) 각각 독립 binary 출력
- **조건**: `{AL, RND} × {10%, 20%, 30%}` = 6개
- **기관 수**: 3개 (inst1, inst2, inst3 — 데이터 균등 대표)
- **라운드**: 20 (× 3 epoch = 총 60 epoch)

#### 선별 모드

| `SELECT_MODE` | 설명 |
|---------------|------|
| `committee` (AL) | `global_priority.json` 기반 상위 K% 케이스 선택 |
| `random` (RND) | 무작위 K% 케이스 선택 (baseline) |

```bash
# Stage 2 비교 실험 (6 조건 순차 실행)
bash run_stage2_compare.sh -E 3 -R 20 -W 4 -f 1 -a fedavg
```

출력:
- `states/p2_formal/s2_{al,rnd}{10,20,30}/inst{N}/` — 기관별 모델
- `states/p2_formal/s2_{al,rnd}{10,20,30}/global/` — 집계 모델
- `results/p2_formal_stage2_compare.txt` — Dice 비교 요약

#### 비교 요약 형식

```
Condition       avg_dice      wt      tc      et   inst1  inst2  inst3
────────────────────────────────────────────────────────────────────
AL-10%            0.7432  0.8210  0.7100  0.6987  ...
RND-10%           0.7105  0.7950  0.6820  0.6545  ...
AL-20%            0.7651  ...
RND-20%           0.7280  ...
AL-30%            0.7820  ...
RND-30%           0.7490  ...
```

---

## 전체 실험 파이프라인

### 로컬 단일 노드 실행

```bash
bash run_e2e.sh \
  -E 60 \        # Stage 1 에폭
  -R 20 \        # Stage 2 라운드
  -W 4  \        # GPU 수 (DDP)
  -f 10 \        # 검증 주기
  -a fedwavg     # FL 알고리즘
```

### Argo Workflow 분리 실행

```bash
# Step 1: Stage 1
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/Stage1_partition2.yaml \
  -p job-prefix=stage1_p2_250327 -n dwnkim

# Step 2: Committee (Stage 1 완료 후)
# argo에는 committee용 workflow가 없으므로 로컬 또는 별도 k8s Job으로 실행:
bash run_committee_stage1.sh -k texture_bald

# Step 3: Stage 2
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/Stage2_partition2.yaml \
  -p job-prefix=stage2_p2_250327 -n dwnkim
```

> Committee step은 Stage 1 모델 경로(PVC 내)를 직접 참조하므로
> `fedpod-states-pvc`가 마운트된 노드(gn144)에서 실행하거나
> 별도 Argo Job으로 감싸서 실행합니다.

---

## 파라미터 파일 위치

```
experiments/params/
  Stage1_partition2.yaml   ← Stage 1 Argo 파라미터
  Stage2_partition2.yaml   ← Stage 2 Argo 파라미터
  FedAVG_partition2.yaml   ← FedAVG 비교 실험
  FedPOD_partition2.yaml   ← FedPOD 비교 실험
```

---

## 데이터 선별 상세

`FeTSDataset`의 `select_mode` / `select_pct` 파라미터로 제어됩니다.

```
run_train.sh -Y committee -y 0.1 -T states/p2_formal/s1/global/global_priority.json
```

| 옵션 | 역할 |
|------|------|
| `-Y committee` | priority 기반 상위 K% 선택 |
| `-Y random` | 무작위 K% 선택 |
| `-y 0.1` | 선택 비율 (10%) |
| `-T path` | priority JSON 경로 |

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `scripts/run_committee_global.py` | K 모델 불확실성 평가 → priority JSON |
| `scripts/run_committee.py` | 단일 기관 committee 평가 |
| `scripts/app.py` | 학습/추론 루프 (select_mode 처리 포함) |
| `scripts/dsets/fets.py` | 데이터셋 (WeightedRandomSampler로 priority 반영) |
| `run_committee_stage1.sh` | Stage 1 이후 committee 자동 실행 |
| `run_stage2_compare.sh` | AL vs RND 6조건 비교 실험 |
| `run_e2e.sh` | Stage1 → Committee → Stage2 일괄 실행 |

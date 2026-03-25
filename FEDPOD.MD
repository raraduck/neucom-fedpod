# FEDPOD.MD — 2-Stage Committee-Guided FL 전략

## 1. 전체 개요

```
Stage 1 (Local Specialization)
  각 기관이 독립적으로 WT 검출 모델을 학습
  → 기관 특화 모델 M1, M2, ..., MK 생성
  (연합학습 없음 — aggregation 없음)

Stage 2 (Committee-Guided FL)
  Stage 1 모델들이 Committee를 구성
  Committee가 훈련 데이터를 평가 → 샘플 우선순위 산출
  우선순위 기반으로 Stage 2 연합학습을 동적으로 수행
```

---

## 2. Stage 1 — 기관 특화 WT 검출

### 2.1 목적

기관별로 데이터 분포가 다르기 때문에, 각 기관이 **자신의 데이터에 특화된 WT 검출 모델**을 보유하는 것이 목표입니다.
연합학습의 평균화로 인한 개인화 손실 없이 기관 고유의 특성을 보존합니다.

### 2.2 학습 설정

```
입력:  [t1, t1ce, t2, flair]  (4채널 MRI)
타깃:  sub.nii.gz → WT = labels 1+2+4 → binary (1채널 출력)
라운드: 1 (단일 라운드)
에폭:  충분한 수 (예: 50~100 epoch)
집계:  없음 — 각 기관 모델 독립 보존
```

### 2.3 실행 흐름

```
기관 1 ─── run_train.sh (R=1, E=100) ──→ states/{job}/inst1/R01r00/models/R01r00_last.pth
기관 2 ─── run_train.sh (R=1, E=100) ──→ states/{job}/inst2/R01r00/models/R01r00_last.pth
기관 K ─── run_train.sh (R=1, E=100) ──→ states/{job}/instK/R01r00/models/R01r00_last.pth

※ run_aggregation.sh 호출 없음
※ 각 기관이 최종 보유: 자신의 데이터에 특화된 WT 모델
```

### 2.4 params 예시 (`Stage1_partition2.yaml`)

```yaml
input_channels: "[t1,t1ce,t2,flair]"
label_groups:   "[[1,2,4]]"
label_names:    "[wt]"
label_index:    "[1]"
rounds:         1
epochs:         100   # 충분한 local 학습
```

---

## 3. Stage 2 — Committee-Guided 연합학습

### 3.1 목적

Stage 1의 기관 특화 모델들이 **Committee**를 구성하여 훈련 데이터의 품질과 학습 우선순위를 평가합니다.
Committee 평가 결과를 기반으로 Stage 2 연합학습에서 샘플을 **동적으로 가중 샘플링**합니다.

### 3.2 학습 설정

```
입력:  [t1, t1ce, t2, flair, seg]  (5채널 — seg는 WT 마스크 proxy, binarize)
타깃:  sub.nii.gz → labels 1, 2, 4 (3채널: NCR, ED, ET)
라운드: N (연합학습)
집계:  FedPOD (P·I·D 기반 가중 집계)
```

### 3.3 Committee 평가 메커니즘

#### Step 1 — Committee 추론

각 기관은 **자신의 Stage 1 모델**을 사용해 로컬 훈련 데이터의 WT 예측을 수행합니다.

```
기관 k의 로컬 훈련 케이스 집합: {x1, x2, ..., xN}

for each case xi:
    pred_wt = M_k(xi)                      # Stage 1 모델로 WT 예측
    gt_wt   = (sub_i ∈ {1,2,4}).float()   # 실제 WT 마스크
    score_i = Dice(pred_wt, gt_wt)         # 케이스별 Committee 점수
```

#### Step 2 — 우선순위 산출

Committee 점수(Dice)가 **중간 범위**에 있는 케이스를 우선 학습합니다.

```
매우 낮은 score (< θ_low):  이상치 또는 어노테이션 오류 가능 → 낮은 가중치
중간 score (θ_low ~ θ_high): 학습 효과가 높은 케이스 → 높은 가중치
매우 높은 score (> θ_high): 이미 잘 아는 케이스 → 낮은 가중치
```

기본 임계값: `θ_low = 0.3`, `θ_high = 0.9`

샘플 가중치 공식:

```
if score < θ_low:   w = score / θ_low          (선형 증가)
if score > θ_high:  w = (1 - score) / (1 - θ_high)  (선형 감소)
else:               w = 1.0                     (최대 가중치)
```

#### Step 3 — 동적 가중 샘플링

Stage 2 DataLoader는 케이스별 가중치를 기반으로 **WeightedRandomSampler**를 사용합니다.
매 FL 라운드 전, Stage 1 모델로 재평가하여 가중치를 갱신합니다.

```
Round 0 시작: Committee 평가 → priority.json 생성
DataLoader:   WeightedRandomSampler(weights=priority)
Round 0 종료: aggregation
Round 1 시작: Committee 재평가 → priority.json 갱신
...
```

### 3.4 전체 실행 흐름

```
[준비] Stage 1 모델 경로 목록 제공

Stage 2 Round r:
  for each 기관 k:
    1. run_committee.sh  ← Stage 1 모델로 로컬 데이터 평가 → priority_{k}.json
    2. run_train.sh      ← priority_{k}.json 기반 WeightedSampler로 학습
  aggregation (FedPOD)
  → 다음 라운드 반복
```

### 3.5 params 예시 (`Stage2_partition2.yaml`)

```yaml
input_channels:  "[t1,t1ce,t2,flair,seg]"
mask_channels:   "[seg]"
label_groups:    "[[1],[2],[4]]"
label_names:     "[ncr,ed,et]"
label_index:     "[1,2,4]"
rounds:          3
epochs:          30
# committee 관련
stage1_model_dir: "states/stage1_p2"   # 기관별 Stage 1 모델 루트
priority_low:     0.3
priority_high:    0.9
```

---

## 4. 구현 로드맵

### 현재 구현 완료

- [x] Stage 1 로컬 학습 (rounds=1, epochs=N)
- [x] Stage 2 5채널 입력 (`mask_channels=[seg]`)
- [x] Stage 2 3클래스 출력 (NCR/ED/ET)

### 구현 필요

| 컴포넌트 | 설명 | 파일 |
|---------|------|------|
| `run_committee.sh` | Stage 1 모델로 로컬 데이터 평가 → priority JSON 생성 | `run_committee.sh` |
| `scripts/run_committee.py` | 케이스별 Dice 계산 & 가중치 산출 | `scripts/run_committee.py` |
| `FeTSDataset` 가중 샘플링 | `priority_path` 인자 → WeightedRandomSampler | `scripts/dsets/fets.py` |
| `args.py` | `--priority_path`, `--priority_low`, `--priority_high` | `scripts/utils/args.py` |
| `run_train.sh` | `-P` 플래그 (priority_path) | `run_train.sh` |

---

## 5. 로컬 테스트 계획

### Step 1 — Stage 1 로컬 학습 (2기관, 1라운드, 충분한 epoch)

```bash
# init
bash run_init.sh -A stage1_local/global -b residual \
  -l "[32,64,128,256]" -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]" -s states

# inst1 (aggregation 없음)
bash run_train.sh -S 1001 -W 4 -J stage1_local/inst1 \
  -R 1 -r 0 -E 50 -i 1 \
  -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]" -N "[wt]" -I "[1]" \
  -M states/stage1_local/global/init.pth ...

# inst2 (aggregation 없음)
bash run_train.sh -S 1002 -W 4 -J stage1_local/inst2 \
  -R 1 -r 0 -E 50 -i 2 ...
```

### Step 2 — Committee 평가 (구현 후)

```bash
bash run_committee.sh \
  -m states/stage1_local/inst1/R01r00/models/R01r00_last.pth \
  -i 1 -c experiments/partition2/fets_split.csv \
  -D /data/fets128/trainval -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]" \
  -o states/stage1_local/inst1/priority.json
```

### Step 3 — Stage 2 Committee-Guided FL

```bash
bash run_train.sh ... \
  -C "[t1,t1ce,t2,flair,seg]" -X "[seg]" \
  -G "[[1],[2],[4]]" -N "[ncr,ed,et]" \
  -P states/stage1_local/inst1/priority.json ...
```

---

## 6. 기대 효과

| 항목 | 기존 FL | Stage 1+2 Committee |
|------|---------|---------------------|
| 기관 특화성 | 집계로 희석 | Stage 1에서 완전 보존 |
| 데이터 품질 반영 | 없음 | Committee 점수로 동적 반영 |
| 이상치 내성 | 낮음 | 저품질 샘플 자동 하향 가중치 |
| 학습 효율 | 균등 샘플링 | 정보량 높은 샘플 집중 학습 |

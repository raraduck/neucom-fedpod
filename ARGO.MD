# Argo Workflow 실행 가이드

fedpod-new의 Federated Learning 실험을 Argo Workflows로 실행하는 방법을 설명합니다.

---

## 전체 구조

```
argo submit
  └─ round-loop (sequential steps)
       ├─ round-0
       │    ├─ train (inst1, inst2, inst3, inst4, inst5 — 병렬)
       │    └─ aggregate
       ├─ round-1
       │    ├─ train × 5 (agg-0 모델 로드)
       │    └─ aggregate
       └─ round-2
            ├─ train × 5 (agg-1 모델 로드)
            └─ aggregate
```

- **train pod**: 기관 수만큼 병렬 실행 (`withParam` fan-out)
- **aggregate pod**: 모든 train pod 완료 후 순차 실행
- **round**: 이전 round의 aggregate 완료 후 다음 round 시작

---

## 출력 폴더 구조

`job-prefix` 하나가 모든 출력 경로를 결정합니다.
실험 이름 폴더 아래에 기관별 결과가 병렬로 담깁니다.

```
/workspace/
├── states/
│   └── {job-prefix}/              ← 실험 이름 폴더
│       ├── inst1/
│       │   └── R03r01/models/
│       │       ├── R03r01_prev.pth   ← pre-validation 체크포인트
│       │       ├── R03r01_best.pth   ← 학습 중 최적 체크포인트
│       │       └── R03r01_last.pth   ← post-validation + PID 저장
│       ├── inst2/ ...
│       └── global/
│           └── R03r01/models/
│               └── R03r01_agg.pth    ← 집계 결과 모델
└── logs/
    └── {job-prefix}/
        ├── inst1/train.log
        ├── inst2/train.log
        └── global/aggregation.log
```

체크포인트 파일명 규칙: `R{총라운드:02}r{현재라운드:02}_{suffix}.pth`

---

## 사전 요구사항

### 1. Kubernetes PVC

아래 3개 PVC가 클러스터에 미리 생성되어 있어야 합니다.

| PVC 이름 | 마운트 경로 | 용도 |
|---|---|---|
| `fets128-data-pvc` | `/data` (read-only) | 학습 데이터 |
| `fedpod-states-pvc` | `/workspace/states` | 체크포인트 저장 |
| `fedpod-logs-pvc` | `/workspace/logs` | 학습 로그 저장 |

### 2. 컨테이너 이미지

`k8s/argo-workflow.yaml` 내 `<registry>` 자리를 실제 이미지 경로로 교체합니다.

```bash
# 이미지 빌드 및 푸시
docker build -t <registry>/fedpod-new:latest .
docker push <registry>/fedpod-new:latest
```

이미지 하나에 학습(train)과 집계(aggregate) 기능이 모두 포함되며,
Argo에서 step별로 다른 커맨드를 호출합니다.

### 3. ServiceAccount

Argo Workflows 실행용 ServiceAccount가 필요합니다.

```bash
kubectl create serviceaccount argo-workflow -n <namespace>
kubectl create rolebinding argo-workflow \
  --clusterrole=argo-role \
  --serviceaccount=<namespace>:argo-workflow \
  -n <namespace>
```

---

## 실험 파라미터 파일

`experiments/params/` 폴더에 실험별 YAML 파일을 작성합니다.

```
experiments/params/
├── FedPOD_partition2.yaml    ← FedPOD 알고리즘, partition2 데이터
└── FedAVG_partition2.yaml    ← FedAVG baseline
```

### 파라미터 설명

| 키 | 설명 | 예시 |
|---|---|---|
| `job-prefix` | 실험 식별자. 출력 폴더 이름에 사용 | `fedpod_p2_fets128` |
| `algorithm` | FL 알고리즘 | `fedpod` \| `fedavg` \| `fedwavg` \| `fedprox` \| `fedpid` |
| `rounds` | 총 FL 라운드 수 | `3` |
| `epochs` | 라운드당 로컬 epoch 수 | `30` |
| `eval_freq` | validation 주기 (epoch 단위) | `5` |
| `data_root` | 학습 데이터 루트 경로 | `/data/fets128/trainval` |
| `split_csv` | 기관 분할 CSV 경로 | `experiments/partition2/fets_split.csv` |
| `input_channels` | 입력 MRI 모달리티 | `"[t1,t1ce,t2,flair]"` |
| `label_groups` | 레이블 그룹핑 | `"[[1,2,4]]"` (WT = NCR+ED+ET) |
| `label_names` | 레이블 이름 | `"[wt]"` |
| `label_index` | 사용할 레이블 인덱스 | `0` |
| `init_model` | 초기 모델 경로 (`none` = 랜덤) | `none` |
| `block` | UNet 블록 유형 | `residual` \| `plain` |
| `channels` | UNet 채널 수 | `"[32,64,128,256]"` |
| `institutions` | 참여 기관 목록 (JSON 배열) | 아래 참조 |

`institutions` 배열 필드:

| 필드 | 설명 |
|---|---|
| `id` | CSV의 `Partition_ID` (폴더 이름과 무관) |
| `seed` | 재현성 시드 |
| `data_pct` | 학습 데이터 사용 비율 (불균형 시뮬레이션) |
| `gpu` | pod당 요청 GPU 수 (DDP 적용) |

> `agg_name`은 `{job-prefix}_global`로 자동 결정되므로 별도 설정 불필요합니다.

---

## 실행 방법

### 기본 실행

```bash
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/FedPOD_partition2.yaml \
  -p job-prefix=fedpod_p2_250325 \
  -n <namespace>
```

`-p job-prefix=<실험명>`으로 전달한 이름이 모든 출력 폴더의 prefix가 됩니다.

### 알고리즘 변경

```bash
# FedAVG로 실행
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/FedAVG_partition2.yaml \
  -p job-prefix=fedavg_p2_250325 \
  -n <namespace>
```

### 파라미터 일부만 override

```bash
# params YAML 기준에서 rounds와 epochs만 변경
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/FedPOD_partition2.yaml \
  -p job-prefix=fedpod_p2_quick \
  -p rounds=1 \
  -p epochs=5 \
  -n <namespace>
```

### 워크플로우 상태 확인

```bash
# 실행 중인 워크플로우 목록
argo list -n <namespace>

# 특정 워크플로우 상태
argo get fedpod-xxxxx -n <namespace>

# 실시간 로그 (pod 이름 필요)
argo logs fedpod-xxxxx -n <namespace> --follow

# 특정 step 로그
argo logs fedpod-xxxxx -n <namespace> -c main --node-field-selector displayName=train
```

### 워크플로우 중단

```bash
argo stop fedpod-xxxxx -n <namespace>
argo delete fedpod-xxxxx -n <namespace>
```

---

## 새 알고리즘 실험 추가

1. `experiments/params/` 에 새 YAML 파일 추가

```yaml
# experiments/params/FedProx_partition2.yaml
job-prefix: fedprox_p2_fets128
algorithm:  fedprox

rounds:     3
epochs:     30
eval_freq:  5

data_root:      /data/fets128/trainval
split_csv:      experiments/partition2/fets_split.csv
input_channels: "[t1,t1ce,t2,flair]"
label_groups:   "[[1,2,4]]"
label_names:    "[wt]"
label_index:    "0"

init_model:     "none"
block:          residual
channels:       "[32,64,128,256]"

institutions:
  - {id: 1, seed: 1001, data_pct: "1.0", gpu: 4}
  - {id: 2, seed: 1002, data_pct: "1.0", gpu: 4}
  - {id: 3, seed: 1003, data_pct: "0.5", gpu: 4}
  - {id: 4, seed: 1004, data_pct: "1.0", gpu: 4}
  - {id: 5, seed: 1005, data_pct: "1.0", gpu: 4}
```

2. 실행

```bash
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/FedProx_partition2.yaml \
  -p job-prefix=fedprox_p2_250325 \
  -n <namespace>
```

---

## Round 수 변경

`k8s/argo-workflow.yaml`의 `round-loop` 섹션에 round 블록을 추가합니다.

```yaml
# Round 3 추가 예시
- - name: round-3
    template: one-round
    arguments:
      parameters:
      - name: round
        value: "3"
      - name: model-path
        value: "/workspace/states/{{workflow.parameters.job-prefix}}_global/R{{workflow.parameters.rounds}}r03/models/R{{workflow.parameters.rounds}}r03_agg.pth"
```

그리고 params YAML에서 `rounds: 4`로 변경합니다.

---

## FL 라운드 내부 흐름

각 기관 pod는 아래 순서로 실행됩니다.

```
pod 시작
  │
  ├─ 1. pre_validation
  │     └─ 모델 로드 (이전 round agg.pth 또는 랜덤 초기화)
  │     └─ validation 실행 → pre_loss, pre_metrics
  │     └─ R{R}r{r}_prev.pth 저장
  │
  ├─ 2. training  (epochs회 반복)
  │     └─ DDP: torchrun --nproc_per_node={gpu}
  │     └─ R{R}r{r}_best.pth 저장 (best val 갱신 시)
  │
  └─ 3. post_validation + PID 계산
        └─ validation 실행 → post_loss, post_metrics
        └─ P = |train dataset|
           I = (pre_loss + post_loss) / 2
           D = pre_loss - post_loss  (≥ 0)
        └─ R{R}r{r}_last.pth 저장 (model + PID 포함)

모든 기관 pod 완료
  │
  └─ aggregate pod 시작
        └─ 각 기관의 _last.pth 로드
        └─ algorithm에 따른 가중치 계산
        └─ R{R}r{next}_agg.pth 저장
        └─ metrics.json 업데이트
```

### FedPOD 가중치 공식

```
W_k = α·(P_k/ΣP) + β·(I_k/ΣI) + γ·(D_k/ΣD)

기본값: α=0.2, β=0.1, γ=0.7
폴백 (D=0): α=0.8, β=0.2  (데이터 크기 + 품질만 반영)
```

| 항 | 의미 |
|---|---|
| P (Proportional) | 학습 데이터 크기 |
| I (Integral) | 평균 손실 (낮을수록 높은 가중치) |
| D (Derivative) | 학습 개선량 (높을수록 높은 가중치) |

---

## 로컬 디버깅 (Argo 없이)

pod 단위 동작을 로컬에서 검증할 때 사용합니다.

```bash
# Round 0, 기관 1, GPU 4개 DDP
bash run_train.sh \
  -W 4 \
  -S 1001 \
  -J fedpod_debug/inst1 \
  -R 3 -r 0 \
  -E 5 \
  -i 1 \
  -a fedpod \
  -M none

# Round 0 집계
bash run_aggregation.sh \
  -j "fedpod_debug/inst1 fedpod_debug/inst2" \
  -A fedpod_debug/global \
  -R 3 -r 0 \
  -a fedpod

# 결과 확인
ls states/fedpod_debug/
#  inst1/  inst2/  global/
```

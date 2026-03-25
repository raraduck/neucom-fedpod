# POD.MD — Pod 기반 FL 학습 구조 설명

## 1. 코드 흐름 설명

### 1.1 Round 단위 실행 개념

FedPOD는 **pod 1개 = FL 1 round** 구조로 동작합니다.
각 기관(institution)은 독립된 pod에서 실행되며, round가 끝나면 서버가 각 pod의 결과를 집계합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│  FL Round r  (pod inst_01 실행)                                  │
│                                                                  │
│  1. pre_validation                                               │
│     - 전달받은 global model로 local val set 평가                   │
│     - pre_loss, pre_dice 기록                                     │
│     - R{R}r{r}_prev.pth 저장 (baseline)                          │
│                                                                  │
│  2. local training  (epoch 1 ~ N)                                │
│     - local train set으로 N epoch 학습                             │
│     - eval_freq마다 중간 val → R{R}r{r}_best.pth 갱신 (선택)       │
│                                                                  │
│  3. post_validation                                              │
│     - 학습 후 동일 val set 재평가                                   │
│     - post_loss, post_dice 기록                                   │
│                                                                  │
│  4. PID 계산 → R{R}r{r}_last.pth 저장 (aggregator가 읽는 파일)     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 PID 값 계산 방식

`_last.pth`에 저장되는 세 가지 지표로 aggregator의 가중치를 결정합니다.

```python
pre_loss  = pre_metrics['total']   # round 시작 시점의 val loss
post_loss = post_metrics['total']  # round 종료 시점의 val loss
post_loss = min(post_loss, pre_loss)  # 안전장치: 학습이 악화되면 D=0

P = len(train_dataset)             # 전체 local 학습 데이터 수
I = (pre_loss + post_loss) / 2.0   # 평균 품질 (낮을수록 좋음)
D = pre_loss - post_loss           # 개선량 (높을수록 많이 개선됨, >= 0)
```

| 지표 | 의미 | 집계 활용 |
|------|------|-----------|
| **P** | 데이터 볼륨 | FedWAVG, FedPOD — 데이터 많은 기관 가중치 증가 |
| **I** | 평균 품질 지표 | FedPOD — 모델 품질 반영 (작을수록 유리) |
| **D** | 학습 개선량 | FedPOD — 실제로 개선된 기관에 높은 가중치 |

**알고리즘별 가중치 공식:**

```
FedAVG:   W_k = 1 / K                              (균등)
FedWAVG:  W_k = P_k / ΣP                           (데이터량 비례)
FedPOD:   W_k = α·(P_k/ΣP) + β·(I_k⁻¹/ΣI⁻¹) + γ·(D_k/ΣD)
                α=0.2, β=0.1, γ=0.7  (D=0이면 α=0.8, β=0.2로 fallback)
```

### 1.3 체크포인트 구조

```
states/
└── {job_name}/
    └── R{rounds:02}r{round:02}/
        └── models/
            ├── R{R}r{r}_prev.pth   ← pre-val 결과 (baseline 기록용)
            ├── R{R}r{r}_best.pth   ← mid-round 최고 Dice (선택적)
            └── R{R}r{r}_last.pth   ← aggregator 입력 (model + PID)
```

**`_last.pth` 내부 구조:**
```python
{
    'model':        OrderedDict,    # model.state_dict()
    'args':         dict,           # 학습에 사용된 전체 인자
    'pre_metrics':  {'total': float, 'dice': float},
    'post_metrics': {'total': float, 'dice': float},
    'P':            int,            # 학습 데이터 수
    'I':            float,          # (pre_loss + post_loss) / 2
    'D':            float,          # pre_loss - post_loss
    'time':         float,          # 학습 소요 시간 (초)
}
```

### 1.4 DDP (Multi-GPU) 처리 흐름

단일 pod 내 4× GPU를 `torchrun`으로 병렬화합니다.
pre/post validation과 checkpoint 저장은 **rank 0만** 담당합니다.

```
torchrun --nproc_per_node=4 scripts/run_train.py ...
         │
         ├─ rank 0 (cuda:0)  ← 로그 출력, val 실행, checkpoint 저장
         ├─ rank 1 (cuda:1)  ← train epoch만 참여
         ├─ rank 2 (cuda:2)  ← train epoch만 참여
         └─ rank 3 (cuda:3)  ← train epoch만 참여

동기화 포인트 (dist.barrier):
  barrier → pre_val(rank0) → barrier
  for epoch:
      train_epoch(all ranks, DistributedSampler로 데이터 분할)
      barrier → mid-val(rank0, eval_freq마다) → barrier
  barrier → post_val(rank0) → PID save(rank0) → barrier
  destroy_process_group
```

DistributedSampler는 train 데이터만 분할하며, `P`는 분할 전 **전체 데이터 수**를 기록합니다.

---

## 2. Docker 이미지 빌드

### 2.1 디렉토리 구조

```
fedpod-new/                         ← 이미지에 포함되는 코드
├── Dockerfile
├── run_train.sh
├── scripts/
│   ├── run_train.py
│   ├── app.py
│   ├── models/
│   ├── dsets/
│   └── utils/
└── experiments/
    └── partition2/
        └── fets_split.csv          ← Partition_ID 기반 FL 기관 분할 정의

data/ (PVC 마운트, 이미지 미포함)
└── fets128/
    ├── trainval/                   ← 모든 케이스 (train + val), 1251개
    │   ├── FeTS2022_00206/
    │   │   ├── FeTS2022_00206_t1.nii.gz
    │   │   ├── FeTS2022_00206_t1ce.nii.gz
    │   │   ├── FeTS2022_00206_t2.nii.gz
    │   │   ├── FeTS2022_00206_flair.nii.gz
    │   │   └── FeTS2022_00206_sub.nii.gz
    │   └── ...
    └── test/                       ← 홀드아웃 테스트 케이스 (현재 비어있음)

※ FL 기관 분할은 fets_split.csv의 Partition_ID 컬럼으로 정의됩니다.
   data 폴더 내 trainval/test 구분은 용도(학습/평가 vs 홀드아웃)의 차이입니다.
```

### 2.2 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
RUN pip install --no-cache-dir \
    monai==1.3.0 \
    nibabel==5.2.0 \
    pandas==2.2.0 \
    numpy==1.26.4

# 코드 복사 (데이터는 PVC로 마운트)
COPY run_train.sh         /workspace/run_train.sh
COPY scripts/             /workspace/scripts/
COPY experiments/         /workspace/experiments/

RUN chmod +x /workspace/run_train.sh

ENTRYPOINT ["bash", "/workspace/run_train.sh"]
```

### 2.3 빌드 & 푸시

```bash
# 빌드
docker build -t fedpod-train:latest .

# 레지스트리 태깅 및 푸시 (예: Docker Hub 또는 사내 레지스트리)
docker tag fedpod-train:latest <registry>/fedpod-train:v1.0
docker push <registry>/fedpod-train:v1.0
```

---

## 3. Kubernetes Job 매니페스트

### 3.1 단일 기관 학습 Job (Round 0, inst_01)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: fedpod-train-inst01-r0
spec:
  template:
    spec:
      restartPolicy: Never

      containers:
      - name: trainer
        image: <registry>/fedpod-train:v1.0

        # run_train.sh 인자 (ENTRYPOINT에 전달)
        args:
          - "-W" - "4"            # NPROC=4 (GPU 수)
          - "-J" - "fedpod_job"   # job_name
          - "-R" - "3"            # 전체 round 수
          - "-r" - "0"            # 현재 round
          - "-E" - "30"           # epochs per round
          - "-e" - "0"            # epoch offset (resume용, 보통 0)
          - "-i" - "1"            # inst_id
          - "-c" - "experiments/partition2/fets_split.csv"
          - "-M" - "none"         # 초기 round: 랜덤 초기화
          # round > 0 이면: "-M" "/states/fedpod_job/R03r00/models/R03r00_agg.pth"
          - "-D" - "/data/fets128/trainval"
          - "-C" - "[t1,t1ce,t2,flair]"
          - "-G" - "[[1,2,4]]"
          - "-N" - "[wt]"
          - "-a" - "fedpod"
          - "-f" - "5"

        resources:
          limits:
            nvidia.com/gpu: "4"
          requests:
            nvidia.com/gpu: "4"

        volumeMounts:
        - name: data-pvc
          mountPath: /data
          readOnly: true
        - name: states-pvc
          mountPath: /workspace/states
        - name: logs-pvc
          mountPath: /workspace/logs

      volumes:
      - name: data-pvc
        persistentVolumeClaim:
          claimName: fets128-data-pvc
      - name: states-pvc
        persistentVolumeClaim:
          claimName: fedpod-states-pvc
      - name: logs-pvc
        persistentVolumeClaim:
          claimName: fedpod-logs-pvc

      nodeSelector:
        accelerator: nvidia-rtx4090
```

### 3.2 다기관 병렬 실행 (IndexedJob)

여러 기관을 동시에 학습시킬 경우 Kubernetes IndexedJob을 사용합니다.
`JOB_COMPLETION_INDEX`를 inst_id로 매핑합니다.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: fedpod-train-r0
spec:
  completions: 5           # 기관 수 (inst_01 ~ inst_05)
  parallelism: 5           # 전부 동시 실행
  completionMode: Indexed

  template:
    spec:
      restartPolicy: Never

      initContainers:
      # JOB_COMPLETION_INDEX(0-based) → inst_id(1-based) 변환
      - name: init-args
        image: busybox
        command: ["/bin/sh", "-c"]
        args:
          - |
            INST=$((${JOB_COMPLETION_INDEX} + 1))
            echo $INST > /shared/inst_id
        volumeMounts:
        - name: shared
          mountPath: /shared

      containers:
      - name: trainer
        image: <registry>/fedpod-train:v1.0
        command: ["/bin/bash", "-c"]
        args:
          - |
            INST=$(cat /shared/inst_id)
            bash /workspace/run_train.sh \
              -W 4 \
              -J fedpod_job_inst${INST} \
              -R 3 -r 0 -E 30 -e 0 \
              -i ${INST} \
              -c experiments/partition2/fets_split.csv \
              -M none \
              -D /data/fets128/trainval \
              -C "[t1,t1ce,t2,flair]" \
              -G "[[1,2,4]]" -N "[wt]" \
              -a fedpod -f 5

        resources:
          limits:
            nvidia.com/gpu: "4"
          requests:
            nvidia.com/gpu: "4"

        volumeMounts:
        - name: shared
          mountPath: /shared
        - name: data-pvc
          mountPath: /data
          readOnly: true
        - name: states-pvc
          mountPath: /workspace/states
        - name: logs-pvc
          mountPath: /workspace/logs

      volumes:
      - name: shared
        emptyDir: {}
      - name: data-pvc
        persistentVolumeClaim:
          claimName: fets128-data-pvc
      - name: states-pvc
        persistentVolumeClaim:
          claimName: fedpod-states-pvc
      - name: logs-pvc
        persistentVolumeClaim:
          claimName: fedpod-logs-pvc
```

---

## 4. FL Round 전체 흐름

```
Round 0
  ┌─ pod inst_01 ─────────────────┐   ┌─ pod inst_02 ─────────────────┐
  │ pre_val (global init model)   │   │ pre_val (global init model)   │
  │ train 30 epochs               │   │ train 30 epochs               │
  │ post_val → PID                │   │ post_val → PID                │
  │ R03r00_last.pth 저장           │   │ R03r00_last.pth 저장           │
  └───────────────────────────────┘   └───────────────────────────────┘
              ↓ (모든 pod 완료 후)
  ┌─ server aggregation job ──────────────────────────────────────────┐
  │ 각 inst의 R03r00_last.pth 로드                                     │
  │ P, I, D 기반으로 W 계산 (FedPOD 공식)                              │
  │ Σ W_k · model_k → global model                                   │
  │ R03r01_agg.pth 저장 (다음 round의 -M 인자)                         │
  └───────────────────────────────────────────────────────────────────┘

Round 1
  ┌─ pod inst_01 ─────────────────┐   ┌─ pod inst_02 ─────────────────┐
  │ pre_val (aggregated model)    │   │ pre_val (aggregated model)    │
  │ -M R03r01_agg.pth 로드        │   │ -M R03r01_agg.pth 로드        │
  │ train 30 epochs               │   │ train 30 epochs               │
  │ post_val → PID                │   │ post_val → PID                │
  │ R03r01_last.pth 저장           │   │ R03r01_last.pth 저장           │
  └───────────────────────────────┘   └───────────────────────────────┘
              ↓ ...반복...
```

### Round간 모델 전달 방법

```bash
# Round 0 실행 (초기 랜덤 모델)
-r 0 -M none

# Round 1 실행 (Round 0 집계 결과 로드)
-r 1 -M /workspace/states/fedpod_job/R03r00/models/R03r00_agg.pth

# Round 2 실행
-r 2 -M /workspace/states/fedpod_job/R03r01/models/R03r01_agg.pth
```

---

## 5. 환경 변수 참고

| 변수 | 설명 | torchrun 자동 설정 |
|------|------|-------------------|
| `LOCAL_RANK` | 현재 프로세스의 로컬 GPU 번호 (0~N-1) | ✅ |
| `RANK` | 전역 프로세스 번호 | ✅ |
| `WORLD_SIZE` | 전체 프로세스 수 | ✅ |
| `MASTER_ADDR` | 멀티노드 시 마스터 주소 | 수동 설정 필요 |
| `MASTER_PORT` | 멀티노드 시 마스터 포트 | 수동 설정 필요 |
| `NPROC` | GPU 수 (run_train.sh 내부 변수) | `-W` 플래그로 설정 |

`LOCAL_RANK`가 존재하면 DDP 모드로 자동 전환됩니다 (`app.py:40`).
`LOCAL_RANK`가 없으면 단일 GPU 또는 CPU 모드로 실행됩니다.

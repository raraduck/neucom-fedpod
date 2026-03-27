# Argo Workflow — fedpod-new

## 개요

fedpod-new 파이프라인은 **Argo Workflows** 위에서 실행됩니다.
각 FL 단계(Stage 1 학습, Committee, Stage 2 학습)는 별도 Workflow YAML로 정의되며,
모두 동일한 컨테이너 이미지(`fedpod-new:latest`)를 사용합니다.

---

## 이미지 빌드

```bash
# Kaniko Job으로 Harbor 레지스트리에 푸시
kubectl apply -f k8s/kaniko-build-job.yaml -n dwnkim
```

| 항목 | 값 |
|------|----|
| Dockerfile | `/workspace/fedpod-new/Dockerfile` |
| Build context | `dir:///workspace/fedpod-new` (fedpod PVC) |
| 이미지 주소 | `192.168.0.80:30002/dwnkim/fedpod-new:latest` |
| 빌드 노드 | `gn144` (fedpod PVC RWO 마운트 노드) |

---

## PVC 구성

Argo Workflow 실행 전에 클러스터에 아래 PVC가 존재해야 합니다.

| PVC 이름 | 컨테이너 마운트 | 용도 | 접근 |
|----------|----------------|------|------|
| `fets128-data-pvc` | `/data` | MRI 학습 데이터 | ReadOnly |
| `fedpod-states-pvc` | `/workspace/states` | 모델 체크포인트 | ReadWrite |
| `fedpod-logs-pvc` | `/workspace/logs` | 학습 로그 | ReadWrite |

---

## Workflow 파일 목록

| 파일 | 용도 |
|------|------|
| `k8s/argo-workflow.yaml` | 다중 라운드 FL 학습 (일반 범용) |

> Stage 1 → Committee → Stage 2 전체 파이프라인은 `run_e2e.sh` 참고.
> Argo 분리 실행 시 아래 파라미터 파일을 `--parameter-file`로 전달.

---

## Workflow 제출

```bash
# Stage 1 (WT 탐지, 4채널 입력)
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/Stage1_partition2.yaml \
  -p job-prefix=stage1_p2_$(date +%y%m%d) \
  -n dwnkim

# Stage 2 (하위 영역 분할, 5채널 입력)
argo submit k8s/argo-workflow.yaml \
  --parameter-file experiments/params/Stage2_partition2.yaml \
  -p job-prefix=stage2_p2_$(date +%y%m%d) \
  -n dwnkim
```

---

## Workflow 구조 (argo-workflow.yaml)

```
round-loop (sequential steps)
  │
  ├─ init             run_init.sh → states/{job-prefix}/global/init.pth
  │
  ├─ round-0
  │    ├─ train (×N)  run_train.sh  (withParam: institutions → fan-out)
  │    └─ aggregate   run_aggregation.sh
  │
  ├─ round-1
  │    ├─ train (×N)
  │    └─ aggregate
  │
  └─ round-2
       ├─ train (×N)
       └─ aggregate
```

각 step에서 실행되는 shell 스크립트:

| step | 진입점 | 출력물 |
|------|--------|--------|
| `init` | `bash /workspace/run_init.sh` | `states/{prefix}/global/init.pth` |
| `train` | `bash /workspace/run_train.sh` | `states/{prefix}/inst{id}/R{R}r{r}/models/` |
| `aggregate` | `bash /workspace/run_aggregation.sh` | `states/{prefix}/global/R{R}r{r+1}/models/*_agg.pth` |

---

## 파라미터 전달

모든 파라미터는 Argo `arguments.parameters`를 통해 컨테이너 args로 전달됩니다.
실험별 설정은 `experiments/params/*.yaml`에 관리합니다.

```
argo submit ... --parameter-file experiments/params/Stage1_partition2.yaml
                             ↓
                   Argo arguments.parameters
                             ↓
          container args: bash /workspace/run_train.sh -J ... -E ... -G ...
                             ↓
                   scripts/run_train.py (Python argparse)
```

### Container 간 출력물 전달

Container 간 파일 전달은 공유 PVC(`fedpod-states-pvc`)를 통해 이루어집니다.

```
[init container]
  → /workspace/states/{prefix}/global/init.pth  (PVC 기록)

[train container (inst1)]
  ← /workspace/states/{prefix}/global/init.pth  (PVC 읽기)
  → /workspace/states/{prefix}/inst1/R01r00/models/R01r00_last.pth

[train container (inst2)]
  ← 동일 init.pth
  → /workspace/states/{prefix}/inst2/R01r00/models/R01r00_last.pth

[aggregate container]
  ← /workspace/states/{prefix}/inst*/R01r00/models/*_last.pth
  → /workspace/states/{prefix}/global/R01r01/models/R01r01_agg.pth

[next round train container]
  ← /workspace/states/{prefix}/global/R01r01/models/R01r01_agg.pth
  ...
```

---

## 출력물 경로 구조

```
/workspace/states/{job-prefix}/
  global/
    init.pth                              ← 라운드 0 공유 초기 모델
    R{R}r{r+1}/models/R{R}r{r+1}_agg.pth ← 집계 모델
  inst{id}/
    R{R}r{r}/models/
      R{R}r{r}_last.pth                  ← 마지막 에폭
      R{R}r{r}_best.pth                  ← 최적 검증 모델

/workspace/logs/{job-prefix}/
  inst{id}/train.log
  global/aggregation.log
```

---

## 서비스 계정

```yaml
serviceAccountName: argo-workflow
```

Namespace `dwnkim`에 아래 RBAC 권한이 필요합니다:
- `pods`, `pods/log` — get, list, watch
- `workflows` — create, get, list, watch, update, patch

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `init.pth not found` | init step 미완료 또는 PVC 마운트 경로 불일치 | `model-path` 파라미터 및 PVC claimName 확인 |
| GPU OOM | batch_size × patch_size 과다 | `-B 1 -H 96` 등으로 줄임 |
| `torchrun` NCCL 오류 | 멀티-GPU DDP 시 pod 간 통신 | 단일 pod 내 multi-GPU로만 사용 (`-W 4`) |
| 이미지 pull 실패 | Harbor insecure registry | `imagePullPolicy: Always` + `insecure: true` 확인 |

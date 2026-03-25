# FedpodFAL.MD — Federated Active Learning 전략

## 1. 개요

FedPOD의 2단계 파이프라인에 **Active Learning (AL)** 을 도입하여 Stage 2 연합학습의 데이터 효율성을 높입니다.

```
[Stage 1] 기관 특화 WT 모델 학습 (local, no aggregation)
    ↓
[Global Committee] 모든 기관의 점수를 통합 → 글로벌 랭킹
    ↓
[Stage 2] 상위 K% 케이스만 학습 (Static AL)
         vs
         랜덤 K% 케이스 학습 (Random Baseline)
```

### 명명 규칙

| 방법 | 이름 | 설명 |
|------|------|------|
| 글로벌 Committee 상위 K% 선택 | **Static AL** (Cold Start AL) | 학습 전 1회 평가, 이후 고정 |
| 랜덤 K% 선택 | **Random Baseline** | seed 고정 무작위 부분 집합 |
| 라운드마다 재평가·재선택 | Dynamic AL *(추후 구현)* | 학습 중 우선순위 갱신 |

---

## 2. 배경 — 왜 글로벌 우선순위가 필요한가

### 기존 로컬 우선순위의 한계

현재 `run_committee.py`는 **기관별로 독립적으로** 우선순위를 계산합니다.

```
기관 1: 자기 케이스 163개 중 상위 30% → 49개 선택
기관 2: 자기 케이스 163개 중 상위 30% → 49개 선택
```

이 방식은 다음 문제를 가집니다:
- 기관 1의 "상위 30%"와 기관 2의 "상위 30%"의 절대적 품질이 다를 수 있음
- 각 기관이 같은 비율로 참여하므로 데이터 품질 불균형이 무시됨
- 글로벌 관점에서 어떤 케이스가 진짜 어렵고 정보량이 높은지 파악 불가

### 글로벌 우선순위의 이점

```
전체 케이스 326개 (기관1: 163, 기관2: 163)를 단일 점수로 랭킹

Global Top 10% → 33개: 기관1 15개 + 기관2 18개  (비율이 달라도 됨)
Global Top 20% → 65개: 기관1 28개 + 기관2 37개
Global Top 30% → 98개: 기관1 42개 + 기관2 56개
```

- 데이터 품질이 높은 기관이 더 많은 케이스로 기여
- 기관 간 비교 가능한 단일 척도
- Random Baseline과 공정한 비교 가능

---

## 3. 방법론 — Global Committee Scoring

### 3.1 평가 프로세스

```
각 기관 k:
  M_k = Stage 1 WT 모델 (기관 k의 로컬 특화 모델)
  for each train case x_i in institution k:
      score_i = Dice(M_k(x_i), GT_WT_i)    ← Stage 1 WT 예측 vs 실제 WT

모든 기관의 (case_id, score) 쌍을 통합
→ 전체 케이스를 score 내림차순으로 정렬
→ global_priority.json 저장
```

### 3.2 케이스 선택 모드

#### Static AL (Committee Mode)
```
global_priority.json 로드
train_cases 중 global 상위 K%에 속하는 케이스만 유지
나머지 케이스는 학습에서 제외 (hard filter)
```

#### Random Baseline
```
train_cases를 seed로 shuffle
상위 K%만 유지 (동일한 절대 수량)
committee 점수와 무관한 무작위 선택
```

### 3.3 global_priority.json 형식

```json
{
  "all_cases": ["FeTS2022_00206", "FeTS2022_01328", ...],
  "all_scores": [0.951, 0.934, ...],
  "institutions": {
    "1": {
      "cases":  ["FeTS2022_00206", ...],
      "scores": [0.951, ...]
    },
    "2": {
      "cases":  ["FeTS2022_01328", ...],
      "scores": [0.934, ...]
    }
  },
  "config": {
    "models": {
      "1": "states/stage1_local/inst1/R01r00/models/R01r00_last.pth",
      "2": "states/stage1_local/inst2/R01r00/models/R01r00_last.pth"
    },
    "input_channels":  "[t1,t1ce,t2,flair]",
    "label_groups":    "[[1,2,4]]"
  }
}
```

`all_cases` / `all_scores`는 **score 내림차순** 정렬 (index 0 = 최고점).

---

## 4. 실험 설계

### 4.1 비교 조건 (6개)

| 조건 | select_mode | select_pct | 케이스 수 (예) |
|------|------------|------------|--------------|
| AL-10 | committee | 0.10 | 전체의 10% |
| AL-20 | committee | 0.20 | 전체의 20% |
| AL-30 | committee | 0.30 | 전체의 30% |
| RND-10 | random | 0.10 | 전체의 10% |
| RND-20 | random | 0.20 | 전체의 20% |
| RND-30 | random | 0.30 | 전체의 30% |

각 조건은 동일한 seed, 동일한 Stage 2 하이퍼파라미터 (rounds, epochs, algorithm)를 사용합니다.

### 4.2 가설

```
AL-K > RND-K  (같은 K%에서 Committee 선택이 Random보다 우수)
AL-30 ≥ RND-100  (30%의 Committee 선택이 전체 Random 학습과 경쟁)
```

### 4.3 평가 지표

- Validation Dice (전체, NCR/ED/ET 각각)
- 학습 케이스 수 대비 Dice (데이터 효율성)

---

## 5. 구현 설계

### 5.1 새로운 파일

#### `run_committee_global.sh` / `scripts/run_committee_global.py`

```
입력:
  --inst_models   "1:path1.pth 2:path2.pth"  ← 기관ID:모델경로 쌍
  --cases_split   CSV 경로
  --data_root     데이터 루트
  --input_channels "[t1,t1ce,t2,flair]"
  --label_groups  "[[1,2,4]]"
  --output_path   global_priority.json 저장 경로
  --use_gpu       1

출력:
  global_priority.json
    - all_cases: 전체 케이스 (score 내림차순)
    - all_scores: 대응 점수
    - institutions: 기관별 breakdown
    - config: 사용된 모델 경로 등 메타정보
```

### 5.2 수정 파일

#### `scripts/utils/args.py`
```python
--select_pct   float  default=1.0
               # Stage 2 학습에 사용할 케이스 비율 (0 < x <= 1.0)

--select_mode  str    default='all'  choices=['all', 'committee', 'random']
               # all:       전체 케이스 사용
               # committee: global_priority.json 기준 상위 K% 선택
               # random:    seed 기반 랜덤 K% 선택
```

#### `scripts/app.py` `run_train()`
```python
# train_cases 로드 후:
train_cases = select_cases(
    cases       = train_cases,
    select_mode = args.select_mode,
    select_pct  = args.select_pct,
    priority_path = args.priority_path,
    seed        = args.seed,
)
# → 기존 data_pct 처리보다 앞에 적용
```

#### `run_train.sh`
```bash
-y  select_pct   (default 1.0)
-Y  select_mode  (default all)
```

### 5.3 select_cases 로직

```python
def select_cases(cases, select_mode, select_pct, priority_path, seed):
    if select_mode == 'all' or select_pct >= 1.0:
        return cases

    n = max(1, int(len(cases) * select_pct))

    if select_mode == 'committee':
        # global_priority.json 의 all_cases 순서 기준 상위 K%
        with open(priority_path) as f:
            gp = json.load(f)
        ranked = gp['all_cases']         # score 내림차순
        top_set = set(ranked[:int(len(ranked) * select_pct)])
        selected = [c for c in ranked if c in set(cases) and c in top_set]
        return selected[:n]

    if select_mode == 'random':
        import random
        rng = random.Random(seed)
        shuffled = list(cases)
        rng.shuffle(shuffled)
        return shuffled[:n]
```

---

## 6. 실행 흐름 (로컬 테스트)

### Step 1 — Stage 1 로컬 학습

```bash
# 기관 1
bash run_train.sh -J stage1_local/inst1 -R 1 -r 0 -E 50 \
  -i 1 -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]" -N "[wt]" -a fedavg \
  -D data/fets128/trainval -c experiments/partition2/fets_split.csv -g 1

# 기관 2
bash run_train.sh -J stage1_local/inst2 -R 1 -r 0 -E 50 \
  -i 2 -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]" -N "[wt]" -a fedavg \
  -D data/fets128/trainval -c experiments/partition2/fets_split.csv -g 1
```

### Step 2 — Global Committee 평가

```bash
bash run_committee_global.sh \
  -m "1:states/stage1_local/inst1/R01r00/models/R01r00_last.pth,2:states/stage1_local/inst2/R01r00/models/R01r00_last.pth" \
  -c experiments/partition2/fets_split.csv \
  -D data/fets128/trainval \
  -C "[t1,t1ce,t2,flair]" -G "[[1,2,4]]" \
  -o states/stage1_local/global_priority.json
```

### Step 3 — Stage 2 Static AL (Committee Top-K%)

```bash
# AL-30: committee 상위 30%
bash run_train.sh -J stage2_al30/inst1 -R 3 -r 0 -E 30 \
  -i 1 -C "[t1,t1ce,t2,flair,seg]" -X "[seg]" \
  -G "[[1],[2],[4]]" -N "[ncr,ed,et]" \
  -T states/stage1_local/global_priority.json \
  -y 0.3 -Y committee \
  -D data/fets128/trainval -c experiments/partition2/fets_split.csv -g 1
```

### Step 4 — Stage 2 Random Baseline (Random K%)

```bash
# RND-30: 랜덤 30%
bash run_train.sh -J stage2_rnd30/inst1 -R 3 -r 0 -E 30 \
  -i 1 -C "[t1,t1ce,t2,flair,seg]" -X "[seg]" \
  -G "[[1],[2],[4]]" -N "[ncr,ed,et]" \
  -y 0.3 -Y random \
  -D data/fets128/trainval -c experiments/partition2/fets_split.csv -g 1
```

---

## 7. 향후 확장 — Dynamic AL

정적 AL은 Stage 1 종료 시 1회만 우선순위를 계산합니다.
Dynamic AL은 매 FL 라운드 전에 **현재 Stage 2 글로벌 모델**로 케이스를 재평가하여 우선순위를 갱신합니다.

```
Dynamic AL 흐름:
  Round r:
    1. 현재 글로벌 모델 M_global^r 로 전체 케이스 평가
    2. 새로운 우선순위 → priority_r.json
    3. Stage 2 각 기관 학습 (WeightedRandomSampler 또는 hard filter)
    4. Aggregation → M_global^{r+1}
    5. 반복
```

Dynamic AL은 정적 AL 실험 결과 검토 후 구현 여부를 결정합니다.

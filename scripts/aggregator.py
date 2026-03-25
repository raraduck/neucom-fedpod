"""Federated aggregation algorithms for fedpod-new.

All algorithms produce a weighted average of local model state dicts.
Only the weight computation differs per algorithm.

Weight formulas:
  fedavg:   W_k = 1/K
  fedwavg:  W_k = P_k / ΣP
  fedprox:  same as fedwavg (proximal term handled at training time)
  fedpod:   W_k = α·(P_k/ΣP) + β·(I_k/ΣI) + γ·(D_k/ΣD)
              where α=0.2, β=0.1, γ=0.7  (fallback α=0.8, β=0.2 when D=0)
  fedpid:   same formula as fedpod but P/I/D are computed across two rounds
              (rounds r-1 and r); falls back to fedwavg for round < 2
"""
import torch


# ── core aggregation (shared by all algorithms) ───────────────────────────────

def aggregate(weights: list[float], models: list[dict]) -> dict:
    """Weighted average of model state dicts.

    Args:
        weights: per-client weights (should sum to ~1.0)
        models:  list of state_dict from each client's _last.pth
    Returns:
        aggregated state_dict
    """
    agg = {k: torch.zeros_like(v, dtype=torch.float)
           for k, v in models[0].items()}
    for w, m in zip(weights, models):
        for k in agg:
            agg[k] += m[k].float() * w
    return agg


# ── weight computation ────────────────────────────────────────────────────────

def compute_weights(algorithm: str,
                    local_states: list[dict],
                    json_history: dict,
                    curr_round: int) -> list[float]:
    """Compute aggregation weights.

    Args:
        algorithm:    one of fedavg | fedwavg | fedprox | fedpod | fedpid
        local_states: list of loaded _last.pth dicts (one per client)
        json_history: round history dict loaded from metrics.json
        curr_round:   current FL round index

    Returns:
        weight list (same length as local_states, sums to 1.0)
    """
    K = len(local_states)

    if algorithm == 'fedavg':
        return [1.0 / K] * K

    P = [float(s['P']) for s in local_states]

    if algorithm in ('fedwavg', 'fedprox'):
        return _normalize(P)

    if algorithm == 'fedpod':
        I = [float(s['I']) for s in local_states]
        D = [float(s['D']) for s in local_states]
        return _pod_weights(P, I, D)

    if algorithm == 'fedpid':
        if curr_round < 2:
            # not enough history: fall back to fedwavg
            return _normalize(P)
        job_names  = [s['args']['job_name'] for s in local_states]
        prev_round = json_history.get(str(curr_round - 1), {})
        this_round = json_history.get(str(curr_round), {})
        loss_prev  = [float(prev_round[j]['post']['total']) for j in job_names]
        loss_curr  = [float(this_round[j]['post']['total']) for j in job_names]
        I = [(a + b) / 2.0 for a, b in zip(loss_prev, loss_curr)]
        D = [max(0.0, a - b)  for a, b in zip(loss_prev, loss_curr)]
        return _pod_weights(P, I, D)

    raise NotImplementedError(f'Unknown algorithm: {algorithm}')


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [v / total for v in values]


def _pod_weights(P: list[float],
                 I: list[float],
                 D: list[float]) -> list[float]:
    """FedPOD weight formula (also used by FedPID with cross-round P/I/D)."""
    sum_P = sum(P)
    sum_I = sum(I)
    sum_D = sum(D)

    if sum_D == 0:
        if sum_I == 0:
            return [1.0 / len(P)] * len(P)
        # D-term collapsed: data-quality blend
        alpha, beta = 0.8, 0.2
        return [alpha * p / sum_P + beta * i / sum_I
                for p, i in zip(P, I)]
    else:
        alpha, beta, gamma = 0.2, 0.1, 0.7
        return [alpha * p / sum_P + beta * i / sum_I + gamma * d / sum_D
                for p, i, d in zip(P, I, D)]

"""Training distributions, fixed evaluation sets and evaluation functions for Stages A, B and C."""
import numpy as np

from .ppo import aggregate, make_envs, rollout
from .systems import stage_a_system, stage_b_system, stage_c_system, with_load

EVAL_SEED = 12345
B_LOADS = [("low", 0.3, False), ("medium", 0.6, False), ("high", 0.85, False), ("bursty", 0.6, True)]


# ---------------- training distributions ----------------
def closed_c_system(rng, slowed=None, n_requests=64):
    """Random Stage C system run in closed loop: one request at a time, all servers idle at every decision,
    so only the history tokens can reveal hidden server speeds."""
    s = stage_c_system(rng, slowed=slowed, n_requests=n_requests, load=0.0)
    s.closed_loop = True
    return s


def train_specs(stage, n_iters, B=16, closed_frac=0.0):
    a, b = stage_a_system(), stage_b_system()

    def make(it, rng):
        if stage == "A":
            return [a for _ in range(B)]
        if stage == "B":
            hi = 0.5 if it < 0.25 * n_iters else 0.95               # curriculum: easy loads first
            return [with_load(b, float(rng.uniform(0.2, hi)), bursty=bool(rng.random() < 0.3)) for _ in range(B)]
        if stage == "C":
            hi = 0.5 if it < 0.25 * n_iters else 0.8
            return [closed_c_system(rng) if rng.random() < closed_frac else
                    stage_c_system(rng, load=float(rng.uniform(0.2, hi))) for _ in range(B)]
        if stage == "H":
            return [closed_c_system(rng) for _ in range(B)]
        raise ValueError(stage)
    return make


# ---------------- fixed evaluation sets ----------------
def eval_set(stage, n=None, seed=EVAL_SEED):
    """List of (spec, seed). Same set for every policy and every seed."""
    rng = np.random.default_rng(seed)
    if stage == "A":
        a = stage_a_system()
        return [(a, int(s)) for s in rng.integers(0, 2 ** 31, size=n or 5)]
    if stage == "B":
        b = stage_b_system()
        out = []
        for name, load, bursty in B_LOADS:
            for s in rng.integers(0, 2 ** 31, size=n or 12):
                out.append((with_load(b, load, bursty), int(s), name))
        return out
    if stage == "C":
        return [(stage_c_system(rng), int(rng.integers(0, 2 ** 31))) for _ in range(n or 48)]
    if stage == "H":
        return [(closed_c_system(rng), int(rng.integers(0, 2 ** 31))) for _ in range(n or 48)]
    raise ValueError(stage)


def adaptation_set(n=24):
    """Stage C adaptation test: the same random systems with and without one GPU secretly 3x slower."""
    out = []
    for k in range(n):
        r1 = np.random.default_rng(EVAL_SEED + 1000 + k)
        r2 = np.random.default_rng(EVAL_SEED + 1000 + k)
        slow = stage_c_system(r1, slowed=True, n_requests=200, load=0.4)
        same = stage_c_system(r2, slowed=False, n_requests=200, load=0.4)
        out.append((slow, same, EVAL_SEED + 5000 + k))
    return out


def model_eval(model, items, env_kw, greedy=True):
    specs = [it[0] for it in items]
    seeds = [it[1] for it in items]
    envs = make_envs(specs, seeds, env_kw)
    rollout(model, envs, greedy=greedy, record=False)
    return envs


def train_eval_fn(stage, env_kw):
    """Small fixed evaluation used during training to pick the best checkpoint."""
    vseed = EVAL_SEED + 777                      # validation systems, separate from the final test sets
    if stage == "A":
        items = eval_set("A", 4, vseed)
    elif stage == "B":
        items = [(s, sd) for s, sd, _ in eval_set("B", 8, vseed)]       # 8 per load = 32 episodes
    elif stage == "C":
        items = eval_set("C", 16, vseed) + eval_set("H", 8, vseed + 1)
    else:
        items = eval_set("H", 24, vseed)

    def fn(model):
        return aggregate(model_eval(model, items, env_kw))
    return fn

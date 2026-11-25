"""Baselines (documentation Section 7.5) and the Stage-A brute-force optimum.

All baselines send requests in earliest-deadline-first order while there is free capacity (they never hold
requests back). Greedy-Predict is *privileged*: it knows the true server speeds, secret slowdowns and outlier
rates, and estimates each candidate server subset by Monte-Carlo with the simulator's own latency model.
"""
import itertools
import math

import numpy as np

from .world import SERVICES, alone_mean_time, size_time_factor


def mc_subset_reward(env, j, subset, n_after, S, rng, waited=None, samples=False):
    """Expected reward of sending request j to `subset` (list of servers), each server running n_after[i]
    copies including this one. Static-load approximation of the simulator's latency model.
    samples=True returns the S reward samples instead of their mean."""
    sp = env.spec
    s = int(env.req_service[j])
    u = int(env.req_user[j])
    W = (env.t - env.arrival[j]) if waited is None else waited
    D1, D2 = env.req_D1[j], env.req_D2[j]
    fin, busy_own = [], []
    for i in subset:
        srv = sp.servers[i]
        net = np.maximum(2.0, sp.net_base[u, i] + rng.normal(0.0, sp.net_jitter, S))
        mean = alone_mean_time(s, srv, size_time_factor(s, env.req_size[j]))
        ex = mean * np.exp(rng.normal(0.0, srv.noise, S) - 0.5 * srv.noise ** 2)
        ex = np.where(rng.random(S) < srv.outlier_p[s], ex * srv.outlier_factor, ex)
        ex = ex * (1.0 + sp.alpha * (n_after[i] - 1))
        fin.append(net / 2 + ex + net / 2)
        busy_own.append(net / 2 + ex)
    fin = np.stack(fin)
    busy_own = np.stack(busy_own)
    win = fin.min(0)
    T = W + win
    P = np.where(T <= D1, 1.0, np.where(T <= D2, (D2 - T) / (D2 - D1), 0.0))
    busy = np.minimum(busy_own, win[None, :]).sum(0)
    r = P - env.w_late * np.maximum(0.0, T - D2) / D2 - env.w_cost * busy / D2
    return r if samples else float(r.mean())


def brute_force_best(env, j, S=4000, seed=0):
    """Stage A: expected reward of every non-empty server subset on an idle system (common random numbers)."""
    n = env.n
    res = {}
    for size in range(1, n + 1):
        for sub in itertools.combinations(range(n), size):
            rng = np.random.default_rng(seed)
            res[sub] = mc_subset_reward(env, j, list(sub), np.ones(n, int), S, rng, waited=0.0)
    return res


# ---------------- heuristic policies: return list of (position in window, server) ----------------
def _loads(env):
    return np.array([len(env.copies_on[i]) for i in range(env.n)])


def policy_random(env, rng, copies=1):
    win = env.window()
    free = env.free_capacity().copy()
    edges = []
    for p, _ in enumerate(win):
        avail = np.flatnonzero(free > 0)
        if len(avail) == 0:
            break
        pick = rng.choice(avail, size=min(copies, len(avail)), replace=False)
        for i in pick:
            edges.append((p, int(i)))
            free[i] -= 1
    return edges


def policy_jsq(env, rng):
    win = env.window()
    free = env.free_capacity().copy()
    load = _loads(env).astype(float)
    edges = []
    for p, _ in enumerate(win):
        avail = np.flatnonzero(free > 0)
        if len(avail) == 0:
            break
        best = avail[load[avail] == load[avail].min()]
        i = int(rng.choice(best))
        edges.append((p, i))
        free[i] -= 1
        load[i] += 1
    return edges


def policy_greedy_predict(env, rng, S=96, max_copies=2):
    """Privileged: for each request (EDF), choose the subset (one server, or two of the 3 best) with the highest
    Monte-Carlo expected reward under the current load, using the true latency model."""
    win = env.window()
    free = env.free_capacity().copy()
    nrun = _loads(env).astype(int)
    edges = []
    for p, j in enumerate(win):
        avail = list(np.flatnonzero(free > 0))
        if not avail:
            break
        cands = []
        singles = []
        for i in avail:
            n_after = nrun + 1
            singles.append((mc_subset_reward(env, j, [i], n_after, S, rng), i))
        singles.sort(reverse=True)
        cands.extend([(r, (i,)) for r, i in singles])
        if min(max_copies, env.d_max) >= 2:
            top = [i for _, i in singles[:3]]
            for a, b in itertools.combinations(top, 2):
                n_after = nrun + 1
                cands.append((mc_subset_reward(env, j, [a, b], n_after, S, rng), (a, b)))
        best = max(cands)[1]
        for i in best:
            edges.append((p, int(i)))
            free[i] -= 1
            nrun[i] += 1
    return edges


def oracle_bound(env, j):
    """Lower bound on latency: request alone on its fastest server, no waiting, mean network time."""
    s = int(env.req_service[j])
    u = int(env.req_user[j])
    best = math.inf
    for i, srv in enumerate(env.spec.servers):
        t = env.spec.net_base[u, i] + alone_mean_time(s, srv, size_time_factor(s, env.req_size[j]))
        best = min(best, t)
    return best

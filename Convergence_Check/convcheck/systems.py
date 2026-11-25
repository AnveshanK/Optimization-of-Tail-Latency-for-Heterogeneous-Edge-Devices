"""Synthetic systems for the three stages of the convergence check (documentation Section 8).

Stage A: one request at a time on a fixed idle system, compared with a brute-force best subset.
Stage B: batches of requests (random and bursty arrivals, several loads) on a fixed system, compared with heuristics.
Stage C: meta-RL. Every episode is a new random system; server speeds are hidden and one GPU server may be
         secretly 3x slower than it looks. Tests whether the model adapts from its history.
"""
import copy
import json

import numpy as np

from .world import ServerSpec, SystemSpec, sample_net


def stage_a_system(seed=11):
    rng = np.random.default_rng(seed)
    servers = [
        ServerSpec("gpu", speed=1.0, capacity=4, outlier_p=(0.15, 0.0, 0.0), noise=0.10),
        ServerSpec("gpu", speed=0.8, capacity=4, outlier_p=(0.15, 0.0, 0.0), noise=0.10),
        ServerSpec("cpu", speed=1.0, capacity=4, outlier_p=(0.02, 0.02, 0.02), noise=0.10),
        ServerSpec("cpu", speed=0.8, capacity=4, outlier_p=(0.02, 0.02, 0.02), noise=0.10),
    ]
    return SystemSpec(servers=servers, n_users=10, net_base=sample_net(rng, 10, 4), closed_loop=True,
                      n_requests=64, load=0.0)


def stage_b_system(seed=22):
    rng = np.random.default_rng(seed)
    servers = [
        ServerSpec("gpu", speed=1.0, capacity=4, outlier_p=(0.05, 0.02, 0.02)),
        ServerSpec("gpu", speed=0.75, capacity=4, outlier_p=(0.05, 0.02, 0.02)),
        ServerSpec("cpu", speed=1.0, capacity=4, outlier_p=(0.02, 0.02, 0.02)),
        ServerSpec("cpu", speed=0.85, capacity=4, outlier_p=(0.02, 0.02, 0.02)),
        ServerSpec("cpu", speed=0.7, capacity=4, outlier_p=(0.02, 0.02, 0.02)),
    ]
    return SystemSpec(servers=servers, n_users=20, net_base=sample_net(rng, 20, 5), n_requests=150, load=0.5)


def with_load(spec, load, bursty=False, n_requests=None):
    s = copy.deepcopy(spec)
    s.load, s.bursty = load, bursty
    if n_requests:
        s.n_requests = n_requests
    return s


def stage_c_system(rng, slowed=None, n_requests=150, load=None, n_range=(3, 6), gpu_range=(1, 2), n_users=20):
    """Random system. slowed: None = random (50%), True = one GPU secretly 3x slower, False = none.
    n_range / gpu_range: inclusive ranges of servers / GPU-like servers (training uses 3-6 / 1-2)."""
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    n_gpu = int(rng.integers(gpu_range[0], gpu_range[1] + 1)) if n > 3 else 1
    servers = []
    for i in range(n):
        dev = "gpu" if i < n_gpu else "cpu"
        if dev == "gpu":
            op = (float(rng.uniform(0.0, 0.15)), float(rng.uniform(0.0, 0.05)), float(rng.uniform(0.0, 0.05)))
        else:
            op = (0.02, 0.02, 0.02)
        servers.append(ServerSpec(dev, speed=float(rng.uniform(0.6, 1.4)), capacity=int(rng.integers(3, 6)),
                                  outlier_p=op))
    if slowed is None:
        slowed = rng.random() < 0.5
    slow_i = int(rng.integers(0, n_gpu))            # drawn always, so the slowed/normal twins stay identical
    if slowed:
        servers[slow_i].slow_factor = 3.0
    order = rng.permutation(n)                      # GPUs are not always first
    servers = [servers[i] for i in order]
    return SystemSpec(servers=servers, n_users=n_users, net_base=sample_net(rng, n_users, n), n_requests=n_requests,
                      load=float(rng.uniform(0.2, 0.8)) if load is None else load,
                      bursty=bool(rng.random() < 0.3))


def spec_to_dict(spec):
    return dict(servers=[s.__dict__ for s in spec.servers], n_users=spec.n_users,
                net_base=np.asarray(spec.net_base).round(3).tolist(), alpha=spec.alpha, load=spec.load,
                bursty=spec.bursty, n_requests=spec.n_requests, closed_loop=spec.closed_loop,
                service_probs=list(spec.service_probs))


def save_spec(spec, path):
    with open(path, "w") as f:
        json.dump(spec_to_dict(spec), f, indent=1)

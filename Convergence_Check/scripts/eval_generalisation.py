"""Generalisation and robustness of the Stage C models (one frozen model, no retraining) under shifts it never saw
in training: larger systems, higher load, tighter deadlines, a skewed service mix. Also decision time at larger size.
Usage: ./run.sh scripts/eval_generalisation.py   -> results/eval/generalisation.json"""
import glob, json, os, time
import numpy as np
import torch

from convcheck.baselines import policy_greedy_predict, policy_jsq, policy_random
from convcheck.obs import build_obs
from convcheck.ppo import aggregate, load_model, make_envs, rollout, run_policy
from convcheck.stages import EVAL_SEED
from convcheck.systems import stage_c_system
from convcheck.world import EdgeEnv

torch.set_num_threads(2)
ENV_KW = dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1)
N = 32


def shifted(kind, k):
    rng = np.random.default_rng(EVAL_SEED + 9000 + k)
    if kind == "in-distribution":
        return stage_c_system(rng)
    if kind == "larger systems (7-8 servers, 2-3 GPUs, 40 users)":
        return stage_c_system(rng, n_range=(7, 8), gpu_range=(2, 3), n_users=40)
    if kind == "higher load (0.9-1.0)":
        return stage_c_system(rng, load=float(rng.uniform(0.9, 1.0)))
    if kind == "tighter deadlines (x0.7)":
        s = stage_c_system(rng); s.deadline_scale = 0.7; return s
    if kind == "skewed service mix (70% speech-like)":
        s = stage_c_system(rng); s.service_probs = (0.7, 0.15, 0.15); return s
    raise ValueError(kind)


KINDS = ["in-distribution", "larger systems (7-8 servers, 2-3 GPUs, 40 users)", "higher load (0.9-1.0)",
         "tighter deadlines (x0.7)", "skewed service mix (70% speech-like)"]
BASE = {"Redundancy-2": lambda e, r: policy_random(e, r, 2), "JSQ": policy_jsq,
        "Greedy-Predict (privileged)": policy_greedy_predict}


def keep(a):
    out = {k: a[k] for k in ["mean_reward", "mean_P", "miss_hard", "p95_norm", "p99_norm", "copies_per_req"]}
    out.update({k: v for k, v in a.items() if k.startswith("audit_")})
    return out


if __name__ == "__main__":
    dirs = sorted(d for d in glob.glob("results/stageC_seed[0-9]") + glob.glob("results/stageC_seed[0-9]_nohist")
                  if os.path.exists(d + "/config.json"))
    res = {}
    for kind in KINDS:
        items = [(shifted(kind, k), EVAL_SEED + 9500 + k) for k in range(N)]
        res[kind] = {}
        for d in dirs:
            model, _ = load_model(d)
            envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
            rollout(model, envs, greedy=True, record=False)
            res[kind][os.path.basename(d)] = keep(aggregate(envs))
        for name, pol in BASE.items():
            envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
            run_policy(pol, envs, np.random.default_rng(7))
            res[kind][name] = keep(aggregate(envs))
        print(kind, {k: round(v["mean_reward"], 3) for k, v in res[kind].items()}, flush=True)
    # decision time at the larger size (8 servers, full window, full history), one thread
    torch.set_num_threads(1)
    model, _ = load_model(dirs[0])
    env = EdgeEnv(stage_c_system(np.random.default_rng(1), n_range=(8, 8), gpu_range=(3, 3), n_users=40, load=0.95,
                                 n_requests=400), np.random.default_rng(1), **ENV_KW)
    ts = []
    while env.advance():
        obs = build_obs(env)
        t0 = time.perf_counter()
        edges, *_ = model.decide([obs], greedy=True)
        ts.append((time.perf_counter() - t0) * 1000)
        env.apply(edges[0], obs["window"])
    res["timing_8_servers"] = dict(mean_ms=float(np.mean(ts)), p99_ms=float(np.percentile(ts, 99)), n=len(ts))
    print("timing 8 servers", res["timing_8_servers"])
    json.dump(res, open("results/eval/generalisation.json", "w"), indent=1)
    print("wrote generalisation.json")

"""History test where ONLY the history tokens can reveal server speeds.

Closed loop (one request at a time, every server idle at each decision, losing copies cancelled), random Stage C
systems with hidden speeds, half of them with one GPU secretly 3x slower. Because the current state is the same at
every decision, a model without history cannot tell fast from slow servers; a model that uses its history can.
Pairs: the same system with and without the slowdown.
Usage: ./run.sh scripts/eval_history.py [run_dir ...]   (default: all results/stageC_* runs)"""
import glob, json, os, sys
import numpy as np
import torch

from convcheck.baselines import policy_greedy_predict, policy_jsq, policy_random
from convcheck.ppo import aggregate, load_model, make_envs, rollout, run_policy
from convcheck.stages import EVAL_SEED
from convcheck.systems import stage_c_system

torch.set_num_threads(4)
ENV_KW = dict(K=8, M=32, d_max=int(os.environ.get("HIST_DMAX", 3)), theta_u=0.25, w_late=1.0, w_cost=0.1)
N_REQ = 96
BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 48), (48, 96)]


def history_set(n=32):
    out = []
    for k in range(n):
        r1 = np.random.default_rng(EVAL_SEED + 3000 + k)
        r2 = np.random.default_rng(EVAL_SEED + 3000 + k)
        slow = stage_c_system(r1, slowed=True, n_requests=N_REQ, load=0.0)
        same = stage_c_system(r2, slowed=False, n_requests=N_REQ, load=0.0)
        for s in (slow, same):
            s.closed_loop = True
        out.append((slow, same, EVAL_SEED + 7000 + k))
    return out


def curves(envs, slowed_idx):
    """Per request-index bucket: mean reward, share of copies on the slowed server."""
    rew, share = [], []
    for lo, hi in BUCKETS:
        rs = [e.req_reward[j] for e in envs for j in range(lo, min(hi, e.R)) if e.req_done[j]]
        rew.append(float(np.mean(rs)))
        tot = hit = 0
        for e, si in zip(envs, slowed_idx):
            for k, i in e.copies_by_decision_server:
                if lo <= k < hi:
                    tot += 1
                    hit += int(i == si)
        share.append(hit / tot if tot else float("nan"))
    return rew, share


def run(policy_name, fn):
    items = history_set()
    slow_specs, same_specs, seeds = [a[0] for a in items], [a[1] for a in items], [a[2] for a in items]
    sidx = [next(i for i, s in enumerate(sp.servers) if s.slow_factor > 1) for sp in slow_specs]
    e_slow, e_same = fn(slow_specs, seeds), fn(same_specs, seeds)
    rs, ss = curves(e_slow, sidx)
    rn, sn = curves(e_same, sidx)
    ag = aggregate(e_slow + e_same)
    return dict(audit={k[6:]: v for k, v in ag.items() if k.startswith("audit_")},
                reward_slowed=aggregate(e_slow)["mean_reward"], reward_normal=aggregate(e_same)["mean_reward"],
                reward_curve_slowed=rs, reward_curve_normal=rn, share_slowed=ss, share_normal=sn,
                copies_slowed=aggregate(e_slow)["copies_per_req"])


def model_fn(d):
    model, _ = load_model(d)
    def fn(specs, seeds):
        envs = make_envs(specs, seeds, ENV_KW)
        rollout(model, envs, greedy=True, record=False)
        return envs
    return fn


def base_fn(pol):
    def fn(specs, seeds):
        envs = make_envs(specs, seeds, ENV_KW)
        run_policy(pol, envs, np.random.default_rng(7))
        return envs
    return fn


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(d for d in glob.glob("results/stageC_seed*") if os.path.exists(d + "/config.json"))
    res = dict(buckets=[f"{a}-{b - 1}" for a, b in BUCKETS], n_systems=len(history_set()), requests=N_REQ, runs={})
    for d in dirs:
        res["runs"][os.path.basename(d)] = run(d, model_fn(d))
        r = res["runs"][os.path.basename(d)]
        print(f"{os.path.basename(d):28s} slowed {r['reward_slowed']:.3f} normal {r['reward_normal']:.3f} "
              f"curve {np.round(r['reward_curve_slowed'], 3)} share {np.round(r['share_slowed'], 3)} "
              f"vs {np.round(r['share_normal'], 3)}", flush=True)
    for name, pol in [("Random-1", lambda e, r: policy_random(e, r, 1)), ("JSQ", policy_jsq),
                      ("Greedy-Predict (privileged)", policy_greedy_predict)]:
        res["runs"][name] = run(name, base_fn(pol))
        r = res["runs"][name]
        print(f"{name:28s} slowed {r['reward_slowed']:.3f} normal {r['reward_normal']:.3f} "
              f"share {np.round(r['share_slowed'], 3)} vs {np.round(r['share_normal'], 3)}", flush=True)
    os.makedirs("results/eval", exist_ok=True)
    out = os.environ.get("HIST_OUT", "results/eval/history_test.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)

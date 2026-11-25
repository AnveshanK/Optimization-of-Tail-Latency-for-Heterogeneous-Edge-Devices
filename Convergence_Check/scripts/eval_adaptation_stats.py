"""Paired statistics for the history / adaptation claims (replaces fixed thresholds by 95% bootstrap intervals).

For every seed, on the same test systems:
  history gain   : reward(with history) - reward(without history), per system      (closed loop, H and C models)
  avoidance gap  : share of copies on the slowed server, unslowed twin minus slowed twin, late in the episode
                   minus at the start of the episode, per twin pair                 (H: closed loop, C: open loop)
A claim passes when the 95% bootstrap interval (10 000 resamples of systems) lies above 0 for every seed.
Usage: ./run.sh scripts/eval_adaptation_stats.py   -> results/eval/adaptation_stats.json"""
import json
import os

import numpy as np
import torch

from convcheck.ppo import load_model, make_envs, rollout
from convcheck.stages import adaptation_set

torch.set_num_threads(4)
B = 10000


def ci(x, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    bs = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=len(x), above_zero=bool(lo > 0))


def per_env_share(e, si, lo, hi):
    tot = hit = 0
    for k, i in e.copies_by_decision_server:
        if lo <= k < hi:
            tot += 1
            hit += int(i == si)
    return hit / tot if tot else np.nan


def run_model(d, specs, seeds, env_kw):
    model, _ = load_model(d)
    envs = make_envs(specs, seeds, env_kw)
    rollout(model, envs, greedy=True, record=False)
    return envs


def closed_loop(prefix, suffix_h, suffix_n, dmax):
    """History gain and avoidance gap on the closed-loop twin set of eval_history.py."""
    os.environ["HIST_DMAX"] = str(dmax)
    import importlib
    import scripts.eval_history as eh
    importlib.reload(eh)
    items = eh.history_set()
    slow, same, seeds = [a[0] for a in items], [a[1] for a in items], [a[2] for a in items]
    sidx = [next(i for i, s in enumerate(sp.servers) if s.slow_factor > 1) for sp in slow]
    out = {}
    for s in range(3):
        res = {}
        for tag, suf in (("hist", suffix_h), ("nohist", suffix_n)):
            d = f"results/{prefix}_seed{s}{suf}"
            es, en = run_model(d, slow, seeds, eh.ENV_KW), run_model(d, same, seeds, eh.ENV_KW)
            res[tag] = dict(rew=np.array([np.mean(e.req_reward) for e in es + en]),
                            gap=np.array([(per_env_share(b, i, 16, 96) - per_env_share(a, i, 16, 96))
                                          - (per_env_share(b, i, 0, 1) - per_env_share(a, i, 0, 1))
                                          for a, b, i in zip(es, en, sidx)]))
        out[f"seed{s}"] = dict(history_gain=ci(res["hist"]["rew"] - res["nohist"]["rew"]),
                               avoidance_gap_hist=ci(res["hist"]["gap"]), avoidance_gap_nohist=ci(res["nohist"]["gap"]))
        print(prefix, s, {k: (round(v["mean"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in out[f"seed{s}"].items()},
              flush=True)
    return out


def open_loop_c():
    """Stage C adaptation set (open loop, d_max = 3): avoidance gap late (decisions 60+) minus early (0-9)."""
    env_kw = dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1)
    ad = adaptation_set()
    slow, same, seeds = [a[0] for a in ad], [a[1] for a in ad], [a[2] for a in ad]
    sidx = [next(i for i, s in enumerate(sp.servers) if s.slow_factor > 1) for sp in slow]
    out = {}
    for s in range(3):
        row = {}
        for tag, suf in (("hist", ""), ("nohist", "_nohist")):
            d = f"results/stageC_seed{s}{suf}"
            es, en = run_model(d, slow, seeds, env_kw), run_model(d, same, seeds, env_kw)
            g = [(per_env_share(b, i, 60, 10 ** 9) - per_env_share(a, i, 60, 10 ** 9))
                 - (per_env_share(b, i, 0, 10) - per_env_share(a, i, 0, 10)) for a, b, i in zip(es, en, sidx)]
            row[f"avoidance_gap_{tag}"] = ci(g)
        out[f"seed{s}"] = row
        print("C open loop", s, {k: (round(v["mean"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in row.items()},
              flush=True)
    return out


if __name__ == "__main__":
    res = dict(H=closed_loop("stageH", "_dmax1_aux", "_nohist_dmax1_aux", 1),
               C_closed_loop=closed_loop("stageC", "", "_nohist", 3),
               C_open_loop=open_loop_c())
    json.dump(res, open("results/eval/adaptation_stats.json", "w"), indent=1)
    print("wrote results/eval/adaptation_stats.json")

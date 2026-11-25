"""Brute-force check of any Stage A run directories (used for the Stage A settings sweep)."""
import sys, json, numpy as np, torch
from convcheck.baselines import brute_force_best
from convcheck.ppo import load_model, make_envs, rollout
from convcheck.stages import eval_set
torch.set_num_threads(2)
items = eval_set("A")
ENV_KW = dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1)
cache = {}
out = {}
for d in sys.argv[1:]:
    for which in ["best.pt", "final.pt"]:
        model, _ = load_model(d, which)
        envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
        rollout(model, envs, greedy=True, record=False)
        ok = []; sp2 = []
        for ei, e in enumerate(envs):
            for j in range(e.R):
                if (ei, j) not in cache: cache[(ei, j)] = brute_force_best(e, j, S=4000, seed=1000 + j)
                t = cache[(ei, j)]; best = max(t.values())
                ch = tuple(sorted(c.server for c in e.req_copies[j])); v = t[ch]
                ok.append(best - v <= max(0.01 * abs(best), 0.005))
                if e.req_service[j] == 0: sp2.append(len(ch) >= 2)
        out[f"{d} {which}"] = dict(within_1pct=float(np.mean(ok)), speech_2plus=float(np.mean(sp2)))
        print(d, which, out[f"{d} {which}"], flush=True)
import os
json.dump(out, open(os.environ.get("A_OUT", "results/eval/stage_A_sweep.json"), "w"), indent=1)

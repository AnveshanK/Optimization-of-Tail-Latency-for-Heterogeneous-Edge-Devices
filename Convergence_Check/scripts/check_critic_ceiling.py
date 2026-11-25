"""Stage A: the highest explained variance any critic can reach.
In Stage A every request is decided alone on an idle system, so the return is the request's own reward. Its noise
(execution noise, outliers, network jitter) cannot be predicted from the state. For the trained policy's choices:
    ceiling = Var_j(E[r | j]) / (Var_j(E[r | j]) + E_j[Var(r | j)])
Usage: ./run.sh scripts/check_critic_ceiling.py   -> results/eval/critic_ceiling.json"""
import json

import numpy as np

from convcheck.baselines import mc_subset_reward
from convcheck.ppo import load_model, make_envs, rollout
from convcheck.stages import eval_set

ENV_KW = dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1)
items = eval_set("A")
out = {}
for s in range(3):
    model, _ = load_model(f"results/stageA_seed{s}")
    envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
    rollout(model, envs, greedy=True, record=False)
    mu, var = [], []
    for e in envs:
        for j in range(e.R):
            sub = sorted(c.server for c in e.req_copies[j])
            r = mc_subset_reward(e, j, sub, np.ones(e.n, int), 4000, np.random.default_rng(j), waited=0.0, samples=True)
            mu.append(r.mean()); var.append(r.var())
    mu, var = np.array(mu), np.array(var)
    out[f"seed{s}"] = dict(n=len(mu), ceiling=float(mu.var() / (mu.var() + var.mean())),
                           var_between=float(mu.var()), var_within=float(var.mean()))
    print(s, out[f"seed{s}"], flush=True)
json.dump(out, open("results/eval/critic_ceiling.json", "w"), indent=1)

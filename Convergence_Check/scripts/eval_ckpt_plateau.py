"""Plateau and seed agreement measured on the full Stage B test set (48 episodes) with the periodic checkpoints of the
last quarter of training, instead of the small validation set used during training.
Usage: ./run.sh scripts/eval_ckpt_plateau.py [tag]    (default tag: none -> results/stageB_seed{s})
      -> results/eval/stageB_ckpt_plateau.json"""
import glob
import json
import os
import sys

import numpy as np
import torch

from convcheck.ppo import aggregate, load_model, make_envs, rollout
from convcheck.stages import eval_set

torch.set_num_threads(4)
ENV_KW = dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1)
tag = sys.argv[1] if len(sys.argv) > 1 else ""
items = eval_set("B")
out = {}
for s in range(3):
    d = f"results/stageB_seed{s}{tag}"
    model, cfg = load_model(d)
    n = cfg["iterations"]
    cks = sorted(glob.glob(os.path.join(d, "ckpt", "it*.pt")))
    cks = [c for c in cks if int(os.path.basename(c)[2:6]) >= 0.75 * n - 1]
    xs, ys = [], []
    for c in cks:
        model.load_state_dict(torch.load(c, map_location="cpu"))
        envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
        rollout(model, envs, greedy=True, record=False)
        xs.append(int(os.path.basename(c)[2:6]))
        ys.append(aggregate(envs)["mean_reward"])
        print(s, xs[-1], round(ys[-1], 4), flush=True)
    slope = float(np.polyfit(xs, ys, 1)[0]) * 100
    out[f"seed{s}"] = dict(iters=xs, test_reward=ys, slope_per_100it=slope, final=ys[-1])
finals = np.array([v["final"] for v in out.values()])
out["final_mean"] = float(finals.mean())
out["final_spread"] = float(finals.max() - finals.min())
out["final_rel_spread"] = float(out["final_spread"] / abs(finals.mean()))
out["plateau"] = all(abs(v["slope_per_100it"]) < 0.05 for k, v in out.items() if k.startswith("seed"))
out["seeds_agree"] = out["final_rel_spread"] <= 0.10 or out["final_spread"] <= 0.03
print({k: v for k, v in out.items() if not k.startswith("seed")})
json.dump(out, open("results/eval/stageB_ckpt_plateau.json", "w"), indent=1)

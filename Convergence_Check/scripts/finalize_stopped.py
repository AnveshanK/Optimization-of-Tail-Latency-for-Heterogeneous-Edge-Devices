"""Write config.json for runs stopped at the time limit (training saves it only at the end), and record where they stopped."""
import glob, json, os
from convcheck.ppo import default_cfg
for d in sorted(glob.glob("results/stage[ABC]_seed[0-9]*")):
    if os.path.exists(os.path.join(d, "config.json")) or not os.path.exists(os.path.join(d, "best.pt")):
        continue
    name = os.path.basename(d)
    stage, seed = name[5], int(name.split("seed")[1].split("_")[0])
    planned = {"A": 120, "B": 300, "C": 400}[stage]
    cfg = default_cfg(name, seed, planned, use_history="nohist" not in name)
    n = sum(1 for _ in open(os.path.join(d, "train_log.csv"))) - 1
    cfg["stopped_at_iteration"] = n
    cfg["note"] = "stopped at the time limit; evaluation uses best.pt (best validation reward so far)"
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    print("finalized", name, "after", n, "iterations")

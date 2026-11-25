"""Train the proposed model for one stage and one seed.  Usage: ./run.sh scripts/train_stage.py A 0 [--no-history] [--iters N]"""
import argparse, os, sys, time
import torch
from convcheck.ppo import Trainer, default_cfg
from convcheck.stages import train_specs, train_eval_fn

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["A", "B", "C", "H"])
ap.add_argument("seed", type=int)
ap.add_argument("--no-history", action="store_true")
ap.add_argument("--iters", type=int, default=None)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--lr", type=float)
ap.add_argument("--lam", type=float)
ap.add_argument("--delta", type=float, help="discount time unit Delta in ms")
ap.add_argument("--ent", type=float)
ap.add_argument("--minibatch", type=int)
ap.add_argument("--lr-final", type=float, help="linear decay of the learning rate to this value")
ap.add_argument("--envs", type=int, help="episodes per iteration (default 16)")
ap.add_argument("--closed-frac", type=float, default=0.0, help="Stage C: share of closed-loop episodes")
ap.add_argument("--dmax", type=int, help="copy limit d_max (default 3)")
ap.add_argument("--aux", type=float, help="weight of the auxiliary speed-prediction loss (training only)")
ap.add_argument("--device", default="cpu", help="cpu or cuda: device of the PPO gradient step (rollouts are always on the CPU)")
ap.add_argument("--ckpt-every", type=int, default=0, help="also save a checkpoint every N iterations (results/<run>/ckpt/)")
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(args.threads)
iters = args.iters or {"A": 300, "B": 400, "C": 400, "H": 200}[args.stage]
name = f"stage{args.stage}_seed{args.seed}" + ("_nohist" if args.no_history else "") + args.tag
cfg = default_cfg(name, args.seed, iters, envs_per_iter=args.envs or 16, use_history=not args.no_history)
cfg["closed_frac"] = args.closed_frac
cfg["device"] = args.device
cfg["ckpt_every"] = args.ckpt_every
if args.dmax:
    cfg["env"]["d_max"] = args.dmax
for k, v in [("lr", args.lr), ("lam", args.lam), ("delta_ms", args.delta), ("ent_coef", args.ent), ("minibatch", args.minibatch),
             ("lr_final", args.lr_final), ("aux_coef", args.aux)]:
    if v is not None:
        cfg["ppo"][k] = v
out = os.path.join("results", name)
tr = Trainer(cfg, out)
t0 = time.time()
tr.train(train_specs(args.stage, iters, cfg["envs_per_iter"], args.closed_frac), train_eval_fn(args.stage, cfg["env"]))
print(f"done {name} in {(time.time() - t0) / 60:.1f} min", flush=True)

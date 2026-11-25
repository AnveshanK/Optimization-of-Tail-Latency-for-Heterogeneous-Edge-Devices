"""Meta-RL training of the proposed scheduler model with PPO (documentation Section 5), plus evaluation helpers.

Rollouts run B simulated systems in parallel; decisions of all systems that are waiting are batched through
the model. Rewards are per request and credited to the decision that sent it; returns are discounted by the
elapsed time (gamma per Delta ms) and advantages use GAE (Eqs. 12-15 of the documentation).
"""
import copy
import csv
import json
import os
import time

import numpy as np
import torch

from .model import SchedulerModel
from .obs import build_obs
from .world import EdgeEnv


class RunningStat:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        for v in np.asarray(x, dtype=np.float64).ravel():
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    @property
    def std(self):
        return max(1e-3, (self.m2 / max(1, self.n - 1)) ** 0.5)


def make_envs(specs, seeds, env_kw):
    return [EdgeEnv(s, np.random.default_rng(int(sd)), **env_kw) for s, sd in zip(specs, seeds)]


def rollout(model, envs, greedy=False, record=True, gen=None, max_steps=None):
    # max_steps = K * d_max + 1 lets a decision use every allowed edge, so the urgent rule holds by construction
    max_steps = max_steps or model.K * model.d_max + 1
    waiting = [e.advance() for e in envs]
    recs = [[] for _ in envs]
    dec_time = []
    while any(waiting):
        idx = [i for i, w in enumerate(waiting) if w]
        obs = [build_obs(envs[i]) for i in idx]
        t0 = time.perf_counter()
        edges, steps, logp, ent, v = model.decide(obs, greedy=greedy, generator=gen, max_steps=max_steps)
        dec_time.append((time.perf_counter() - t0) / len(idx))
        for n, i in enumerate(idx):
            k = envs[i].apply(edges[n], obs[n]["window"])
            if record:
                recs[i].append(dict(obs=obs[n], steps=steps[n], logp=float(logp[n]), v=float(v[n]), k=k,
                                    ent=float(ent[n])))
            waiting[i] = envs[i].advance()
    return recs, float(np.mean(dec_time)) if dec_time else 0.0


def run_policy(policy, envs, rng):
    """Run a baseline policy (function env, rng -> edges) on environments."""
    for e in envs:
        while e.advance():
            win = e.window()
            e.apply(policy(e, rng), win)
    return envs


def aggregate(envs):
    """Pool all finished requests of several episodes into one set of metrics."""
    T, D2, P, R, C, S, miss = [], [], [], [], [], [], []
    cks = []
    for e in envs:
        d = e.req_done
        T.append(e.req_T[d]); D2.append(e.req_D2[d]); P.append(e.req_P[d]); R.append(e.req_reward[d])
        C.append(e.req_ncopies[d]); S.append(e.req_service[d])
        cks.extend([c for c in e.decision_c if c > 0])
    T, D2, P, R, C, S = map(np.concatenate, (T, D2, P, R, C, S))
    Tn = T / D2
    out = dict(mean_reward=float(R.mean()), mean_P=float(P.mean()), miss_hard=float((T > D2).mean()),
               p50_norm=float(np.percentile(Tn, 50)), p95_norm=float(np.percentile(Tn, 95)),
               p99_norm=float(np.percentile(Tn, 99)), copies_per_req=float(C.mean()),
               mean_c=float(np.mean(cks)) if cks else 0.0, n_requests=int(len(T)))
    for s in np.unique(S):
        m = S == s
        out[f"svc{s}_p95_ms"] = float(np.percentile(T[m], 95))
        out[f"svc{s}_p99_ms"] = float(np.percentile(T[m], 99))
        out[f"svc{s}_meanP"] = float(P[m].mean())
    for k in envs[0].audit:
        out["audit_" + k] = int(sum(e.audit[k] for e in envs))
    return out


def compute_advantages(env, recs, vstat, gamma, delta_ms, lam):
    D = len(recs)
    t = np.asarray(env.decision_times)
    R = np.asarray(env.decision_rewards)
    V = np.array([r["v"] for r in recs]) * vstat.std + vstat.mean
    adv = np.zeros(D)
    last = 0.0
    for k in range(D - 1, -1, -1):
        if k == D - 1:
            g, vnext = 0.0, 0.0
        else:
            g = gamma ** ((t[k + 1] - t[k]) / delta_ms)
            vnext = V[k + 1]
        delta = R[k] + g * vnext - V[k]
        last = delta + g * lam * last
        adv[k] = last
    return adv, adv + V, V


class Trainer:
    def __init__(self, cfg, out_dir):
        self.cfg = cfg
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.gen = torch.Generator().manual_seed(cfg["seed"])
        m = cfg["model"]
        self.model = SchedulerModel(d=m["d"], layers=m["layers"], heads=m["heads"], ffn=m["ffn"], mix_hidden=m["mix_hidden"],
                                C=m["C"], d_max=cfg["env"]["d_max"], K=cfg["env"]["K"],
                                use_history=m.get("use_history", True), stop_bias=m.get("stop_bias", 0.0),
                                aux=cfg["ppo"].get("aux_coef", 0.0) > 0)
        # learner = the copy that takes gradient steps (on the GPU if cfg["device"] == "cuda");
        # self.model stays on the CPU for rollouts, validation and checkpoints and is synced after every update
        self.device = torch.device(cfg.get("device", "cpu"))
        self.learner = self.model if self.device.type == "cpu" else copy.deepcopy(self.model).to(self.device)
        self.opt = torch.optim.Adam(self.learner.parameters(), lr=cfg["ppo"]["lr"])
        self.vstat = RunningStat()
        self.log_path = os.path.join(out_dir, "train_log.csv")
        self.best_eval = -1e9

    def update(self, recs_all, adv_all, ret_all):
        p = self.cfg["ppo"]
        obs = [r["obs"] for r in recs_all]
        steps = [r["steps"] for r in recs_all]
        dev = self.device
        old_logp = torch.tensor([r["logp"] for r in recs_all], dtype=torch.float32, device=dev)
        adv = torch.tensor(np.asarray(adv_all), dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.vstat.update(ret_all)
        ret_n = torch.tensor((np.asarray(ret_all) - self.vstat.mean) / self.vstat.std, dtype=torch.float32, device=dev)
        N = len(recs_all)
        stats = dict(pg_loss=[], v_loss=[], entropy=[], approx_kl=[], clipfrac=[], grad_norm=[], aux_loss=[])
        epochs_done = 0
        for ep in range(p["epochs"]):
            perm = self.rng.permutation(N)
            kls = []
            for s0 in range(0, N, p["minibatch"]):
                mb = perm[s0:s0 + p["minibatch"]]
                mbt = torch.as_tensor(mb, device=dev)
                logp, ent, v = self.learner.evaluate([obs[i] for i in mb], [steps[i] for i in mb])
                ratio = torch.exp(logp - old_logp[mbt])
                a = adv[mbt]
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - p["clip"], 1 + p["clip"]) * a).mean()
                vl = ((v - ret_n[mbt]) ** 2).mean()
                loss = pg + p["vf_coef"] * vl - p["ent_coef"] * ent.mean()
                if self.learner.aux is not None:
                    loss = loss + p["aux_coef"] * self.learner.last_aux_loss
                    stats["aux_loss"].append(self.learner.last_aux_loss.item())
                self.opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), p["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    kl = (old_logp[mbt] - logp).mean().item()
                    kls.append(kl)
                    stats["pg_loss"].append(pg.item()); stats["v_loss"].append(vl.item())
                    stats["entropy"].append(ent.mean().item()); stats["approx_kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > p["clip"]).float().mean().item())
                    stats["grad_norm"].append(float(gn))
            epochs_done += 1
            if np.mean(kls) > 1.5 * p["target_kl"]:
                break
        if self.learner is not self.model:
            self.model.load_state_dict({k: v.cpu() for k, v in self.learner.state_dict().items()})
        out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
        out["epochs_done"] = epochs_done
        return out

    def train(self, make_specs, evaluate_fn=None):
        cfg = self.cfg
        B, iters = cfg["envs_per_iter"], cfg["iterations"]
        fields = None
        t_start = time.time()
        lr0, lr1 = cfg["ppo"]["lr"], cfg["ppo"].get("lr_final")
        for it in range(iters):
            t0 = time.time()
            if lr1 is not None:                      # linear learning-rate decay lr -> lr_final
                for g in self.opt.param_groups:
                    g["lr"] = lr0 + (lr1 - lr0) * it / max(1, iters - 1)
            specs = make_specs(it, self.rng)
            seeds = self.rng.integers(0, 2 ** 31, size=B)
            envs = make_envs(specs, seeds, cfg["env"])
            self.model.eval()
            recs, dec_t = rollout(self.model, envs, greedy=False, gen=self.gen)
            t_roll = time.time() - t0
            recs_all, adv_all, ret_all, val_all = [], [], [], []
            for e, rc in zip(envs, recs):
                if not rc:
                    continue
                adv, ret, V = compute_advantages(e, rc, self.vstat, cfg["ppo"]["gamma"], cfg["ppo"]["delta_ms"],
                                                 cfg["ppo"]["lam"])
                recs_all.extend(rc); adv_all.extend(adv); ret_all.extend(ret); val_all.extend(V)
            ret_arr, val_arr = np.asarray(ret_all), np.asarray(val_all)
            ev = 1.0 - np.var(ret_arr - val_arr) / (np.var(ret_arr) + 1e-8)
            self.learner.train()
            ust = self.update(recs_all, adv_all, ret_all)
            agg = aggregate(envs)
            row = dict(iter=it, time_s=round(time.time() - t_start, 1), rollout_s=round(t_roll, 2),
                       iter_s=round(time.time() - t0, 2), decisions=len(recs_all), explained_var=float(ev),
                       train_reward=agg["mean_reward"], train_P=agg["mean_P"], train_miss=agg["miss_hard"],
                       train_p95=agg["p95_norm"], train_copies=agg["copies_per_req"], train_c=agg["mean_c"],
                       decide_ms=dec_t * 1000, **ust)
            if evaluate_fn is not None and (it % cfg["eval_every"] == 0 or it == iters - 1):
                ev_res = evaluate_fn(self.model)
                row["eval_reward"] = ev_res["mean_reward"]
                if ev_res["mean_reward"] > self.best_eval:
                    self.best_eval = ev_res["mean_reward"]
                    torch.save(self.model.state_dict(), os.path.join(self.out, "best.pt"))
            else:
                row["eval_reward"] = ""
            ce = cfg.get("ckpt_every", 0)
            if ce and (it % ce == 0 or it == iters - 1):
                os.makedirs(os.path.join(self.out, "ckpt"), exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.out, "ckpt", f"it{it:04d}.pt"))
            if fields is None:
                fields = list(row.keys())
                with open(self.log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)
            if it % 10 == 0 or it == iters - 1:
                print(f"[{cfg['name']}] it {it:4d}  reward {row['train_reward']:+.3f}  eval {row['eval_reward']}  "
                      f"ent {row['entropy']:.3f}  kl {row['approx_kl']:.4f}  ev {ev:+.2f}  "
                      f"copies {row['train_copies']:.2f}  {row['iter_s']:.1f}s", flush=True)
        torch.save(self.model.state_dict(), os.path.join(self.out, "final.pt"))
        with open(os.path.join(self.out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        return self.model


def default_cfg(name, seed, iterations, envs_per_iter=16, use_history=True):
    return dict(
        name=name, seed=seed, iterations=iterations, envs_per_iter=envs_per_iter, eval_every=10,
        env=dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1),
        model=dict(d=64, layers=2, heads=4, ffn=128, mix_hidden=16, C=10.0, use_history=use_history, stop_bias=0.5),
        # tuned in the pilot runs (documentation starting values: lr 3e-4, minibatch 512, Delta 100 ms, lambda 0.95)
        ppo=dict(lr=1e-3, epochs=4, minibatch=256, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 target_kl=0.02, gamma=0.99, delta_ms=20.0, lam=0.9),
    )


def load_model(run_dir, which=None):
    """Loads the final checkpoint (no checkpoint picking). CONVCHECK_CKPT=best.pt loads the best-validation one instead."""
    which = which or os.environ.get("CONVCHECK_CKPT", "final.pt")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    m = cfg["model"]
    model = SchedulerModel(d=m["d"], layers=m["layers"], heads=m["heads"], ffn=m["ffn"], mix_hidden=m["mix_hidden"],
                       C=m["C"], d_max=cfg["env"]["d_max"], K=cfg["env"]["K"], use_history=m.get("use_history", True),
                       aux=cfg["ppo"].get("aux_coef", 0.0) > 0)
    model.load_state_dict(torch.load(os.path.join(run_dir, which), map_location="cpu"))
    model.eval()
    return model, cfg

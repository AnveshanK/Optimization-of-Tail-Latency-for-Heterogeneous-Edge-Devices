"""Evaluate trained models against the checks of documentation Section 8.
Usage: ./run.sh scripts/evaluate.py [A] [B] [C] [rules] [timing]   (default: all)
Writes JSON files to results/eval/."""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from convcheck.baselines import (brute_force_best, oracle_bound, policy_greedy_predict, policy_jsq, policy_random)
from convcheck.model import SchedulerModel
from convcheck.obs import build_obs
from convcheck.ppo import aggregate, load_model, make_envs, rollout, run_policy
from convcheck.stages import adaptation_set, eval_set
from convcheck.systems import stage_b_system, with_load
from convcheck.world import EdgeEnv

torch.set_num_threads(4)
OUT = "results/eval"
os.makedirs(OUT, exist_ok=True)
ENV_KW = dict(K=8, M=32, d_max=3, theta_u=0.25, w_late=1.0, w_cost=0.1)
SEEDS = [0, 1, 2]


B_TAG = os.environ.get("B_TAG", "")          # e.g. B_TAG=_delta5 evaluates results/stageB_seed{s}_delta5


def runs(stage, nohist=False):
    out = []
    for s in SEEDS:
        d = f"results/stage{stage}_seed{s}" + ("_nohist" if nohist else "") + (B_TAG if stage == "B" else "")
        if os.path.exists(os.path.join(d, "config.json")):
            out.append((s, d))
    return out


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name, flush=True)


BASELINES = {
    "Random-1": lambda e, r: policy_random(e, r, 1),
    "Redundancy-2": lambda e, r: policy_random(e, r, 2),
    "JSQ": policy_jsq,
    "Greedy-Predict (privileged)": policy_greedy_predict,
}


# ------------------------------------------------------------------ Stage A
def eval_stage_a():
    items = eval_set("A")
    bf_cache = {}
    res = dict(models={}, baselines={})

    def score(envs, tag):
        rows = []
        for ei, e in enumerate(envs):
            for j in range(e.R):
                key = (ei, j)
                if key not in bf_cache:
                    bf_cache[key] = brute_force_best(e, j, S=4000, seed=1000 + j)
                table = bf_cache[key]
                best_sub = max(table, key=table.get)
                best = table[best_sub]
                chosen = tuple(sorted(c.server for c in e.req_copies[j]))
                val = table.get(chosen, np.nan)
                rows.append(dict(service=int(e.req_service[j]), best=best, chosen=val, best_sub=list(best_sub),
                                 chosen_sub=list(chosen), regret=best - val))
        reg = np.array([r["regret"] for r in rows])
        best = np.array([r["best"] for r in rows])
        tol = np.maximum(0.01 * np.abs(best), 0.005)
        within = reg <= tol
        exact = np.array([r["best_sub"] == r["chosen_sub"] for r in rows])
        per_svc = {}
        for s in range(3):
            m = np.array([r["service"] == s for r in rows])
            two_best = np.array([len(r["best_sub"]) >= 2 for r in rows])[m]
            two_chosen = np.array([len(r["chosen_sub"]) >= 2 for r in rows])[m]
            per_svc[s] = dict(n=int(m.sum()), within_1pct=float(within[m].mean()), mean_regret=float(reg[m].mean()),
                              bf_says_2plus=float(two_best.mean()), model_2plus=float(two_chosen.mean()),
                              agree_on_copies=float((two_best == two_chosen).mean()))
        return dict(n=len(rows), within_1pct=float(within.mean()), exact=float(exact.mean()),
                    mean_regret=float(reg.mean()), p95_regret=float(np.percentile(reg, 95)),
                    mean_best=float(best.mean()), per_service=per_svc)

    for s, d in runs("A"):
        model, cfg = load_model(d)
        envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
        rollout(model, envs, greedy=True, record=False)
        res["models"][f"seed{s}"] = score(envs, f"seed{s}")
    # untrained model for reference
    torch.manual_seed(0)
    m0 = SchedulerModel(d=64, layers=2, heads=4, ffn=128, mix_hidden=16, stop_bias=0.5)
    envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
    rollout(m0, envs, greedy=True, record=False)
    res["baselines"]["untrained model"] = score(envs, "untrained")
    for name in ["Random-1", "JSQ", "Greedy-Predict (privileged)"]:
        envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
        run_policy(BASELINES[name], envs, np.random.default_rng(7))
        res["baselines"][name] = score(envs, name)
    save("stage_A.json", res)


# ------------------------------------------------------------------ Stage B
def eval_stage_b():
    items = eval_set("B")
    loads = sorted(set(i[2] for i in items), key=["low", "medium", "high", "bursty"].index)
    res = {}
    for load in loads:
        it = [(s, sd) for s, sd, name in items if name == load]
        res[load] = {}
        for s, d in runs("B"):
            model, _ = load_model(d)
            envs = make_envs([x[0] for x in it], [x[1] for x in it], ENV_KW)
            rollout(model, envs, greedy=True, record=False)
            res[load][f"Proposed model seed{s}"] = aggregate(envs)
        for name, pol in BASELINES.items():
            envs = make_envs([x[0] for x in it], [x[1] for x in it], ENV_KW)
            run_policy(pol, envs, np.random.default_rng(7))
            res[load][name] = aggregate(envs)
        # oracle bound (latency only)
        envs = make_envs([x[0] for x in it], [x[1] for x in it], ENV_KW)
        ob = []
        for e in envs:
            ob.extend([oracle_bound(e, j) / e.req_D2[j] for j in range(e.R)])
        res[load]["Oracle bound"] = dict(p50_norm=float(np.percentile(ob, 50)), p95_norm=float(np.percentile(ob, 95)),
                                         p99_norm=float(np.percentile(ob, 99)))
        print("stage B", load, "done", flush=True)
    save(f"stage_B{B_TAG}.json", res)


# ------------------------------------------------------------------ Stage C
def share_curve(envs, slowed_idx, buckets=(0, 10, 30, 60, 120, 10 ** 9)):
    """Share of copies sent to the secretly slowed server, by decision index bucket."""
    out = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        tot = hit = 0
        for e, si in zip(envs, slowed_idx):
            for k, i in e.copies_by_decision_server:
                if lo <= k < hi:
                    tot += 1
                    hit += int(i == si)
        out.append(hit / tot if tot else float("nan"))
    return out


def eval_stage_c():
    items = eval_set("C")
    res = dict(validation={}, adaptation={})
    groups = [("Proposed model (history)", runs("C")), ("Proposed model (no history)", runs("C", nohist=True))]
    for gname, rs in groups:
        for s, d in rs:
            model, _ = load_model(d)
            envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
            rollout(model, envs, greedy=True, record=False)
            res["validation"][f"{gname} seed{s}"] = aggregate(envs)
    for name, pol in BASELINES.items():
        envs = make_envs([i[0] for i in items], [i[1] for i in items], ENV_KW)
        run_policy(pol, envs, np.random.default_rng(7))
        res["validation"][name] = aggregate(envs)
    print("stage C validation done", flush=True)
    ad = adaptation_set()
    slow_specs = [a[0] for a in ad]
    same_specs = [a[1] for a in ad]
    seeds = [a[2] for a in ad]
    slowed_idx = [next(i for i, s in enumerate(sp.servers) if s.slow_factor > 1) for sp in slow_specs]
    for gname, rs in groups:
        for s, d in rs:
            model, _ = load_model(d)
            e_slow = make_envs(slow_specs, seeds, ENV_KW)
            rollout(model, e_slow, greedy=True, record=False)
            e_same = make_envs(same_specs, seeds, ENV_KW)
            rollout(model, e_same, greedy=True, record=False)
            res["adaptation"][f"{gname} seed{s}"] = dict(
                slowed=share_curve(e_slow, slowed_idx), normal=share_curve(e_same, slowed_idx),
                reward_slowed=aggregate(e_slow)["mean_reward"], reward_normal=aggregate(e_same)["mean_reward"])
    for name in ["JSQ", "Greedy-Predict (privileged)"]:
        e_slow = make_envs(slow_specs, seeds, ENV_KW)
        run_policy(BASELINES[name], e_slow, np.random.default_rng(7))
        e_same = make_envs(same_specs, seeds, ENV_KW)
        run_policy(BASELINES[name], e_same, np.random.default_rng(7))
        res["adaptation"][name] = dict(slowed=share_curve(e_slow, slowed_idx), normal=share_curve(e_same, slowed_idx),
                                       reward_slowed=aggregate(e_slow)["mean_reward"],
                                       reward_normal=aggregate(e_same)["mean_reward"])
    res["adaptation_buckets"] = ["0-9", "10-29", "30-59", "60-119", "120+"]
    save("stage_C.json", res)


# ------------------------------------------------------------------ hard rules (masks)
def check_rules():
    torch.manual_seed(0)
    m = SchedulerModel(d=64, layers=2, heads=4, ffn=128, mix_hidden=16)
    with torch.no_grad():                       # make the model want to STOP immediately
        m.stop[-1].bias.fill_(50.0)
        m.mix[-1].bias.fill_(-50.0)
    out = {}
    spec = with_load(stage_b_system(), 0.95)
    # collect decision states from a JSQ run
    states = []
    e = EdgeEnv(spec, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    while e.advance():
        o = build_obs(e)
        states.append(o)
        e.apply(policy_jsq(e, rng), o["window"])
    # 1) urgent request must be sent
    ok = tot = 0
    for o in states:
        if (o["free"] > 0).any():
            o2 = dict(o); o2["urgent"] = o["urgent"].copy(); o2["urgent"][0] = True; o2["idle_any"] = False
            edges, *_ = m.decide([o2], greedy=True)
            tot += 1
            ok += int(any(p == 0 for p, _ in edges[0]))
    out["urgent_request_always_sent"] = dict(passed=ok == tot, cases=tot, ok=ok)
    # 2) idle server + waiting requests -> at least one edge
    ok = tot = 0
    for o in states:
        if (o["free"] > 0).any():
            o2 = dict(o); o2["urgent"] = np.zeros_like(o["urgent"]); o2["idle_any"] = True
            edges, *_ = m.decide([o2], greedy=True)
            tot += 1
            ok += int(len(edges[0]) >= 1)
    out["idle_server_forces_one_edge"] = dict(passed=ok == tot, cases=tot, ok=ok)
    # 3) otherwise holding back is possible (STOP-loving model sends nothing)
    ok = tot = 0
    for o in states:
        o2 = dict(o); o2["urgent"] = np.zeros_like(o["urgent"]); o2["idle_any"] = False
        edges, *_ = m.decide([o2], greedy=True)
        tot += 1
        ok += int(len(edges[0]) == 0)
    out["holding_back_allowed"] = dict(passed=ok == tot, cases=tot, ok=ok)
    # 4) capacity and copy limits never violated (random sampling policy, high load)
    torch.manual_seed(1)
    m2 = SchedulerModel(d=64, layers=2, heads=4, ffn=128, mix_hidden=16)
    viol_cap = viol_copies = viol_dup = 0
    n_dec = 0
    for sd in range(4):
        e = EdgeEnv(spec, np.random.default_rng(sd))
        while e.advance():
            o = build_obs(e)
            edges, *_ = m2.decide([o], greedy=False)
            n_dec += 1
            free = o["free"].copy()
            per_req = {}
            for p, i in edges[0]:
                free[i] -= 1
                per_req[p] = per_req.get(p, 0) + 1
            viol_cap += int((free < 0).any())
            viol_copies += int(any(v > 3 for v in per_req.values()))
            viol_dup += int(len(set(edges[0])) != len(edges[0]))
            e.apply(edges[0], o["window"])
    out["capacity_copy_limits"] = dict(passed=viol_cap == 0 and viol_copies == 0 and viol_dup == 0,
                                       decisions=n_dec, capacity_violations=viol_cap,
                                       copy_limit_violations=viol_copies, duplicate_edges=viol_dup)
    save("rules.json", out)


# ------------------------------------------------------------------ decision time
def timing():
    torch.set_num_threads(1)
    spec = with_load(stage_b_system(), 0.85)
    states = []
    for sd in range(3):
        e = EdgeEnv(spec, np.random.default_rng(sd))
        rng = np.random.default_rng(sd)
        while e.advance():
            o = build_obs(e)
            states.append(o)
            e.apply(policy_jsq(e, rng), o["window"])
    states = states[:400]
    out = {}
    trained = runs("B")
    models = []
    if trained:
        models.append(("trained (d=64, 2 layers)", load_model(trained[0][1])[0]))
    models.append(("documentation size (d=128, 3 layers)", SchedulerModel(d=128, layers=3, heads=4, ffn=256)))
    for name, m in models:
        m.eval()
        ts = []
        for o in states:
            t0 = time.perf_counter()
            m.decide([o], greedy=True)
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts[20:])
        t_obs = []
        for o in states[:200]:
            pass
        out[name] = dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)),
                         p99_ms=float(np.percentile(ts, 99)), n=int(len(ts)))
    # observation building time
    e = EdgeEnv(spec, np.random.default_rng(9))
    rng = np.random.default_rng(9)
    tb = []
    while e.advance():
        t0 = time.perf_counter()
        o = build_obs(e)
        tb.append((time.perf_counter() - t0) * 1000)
        e.apply(policy_jsq(e, rng), o["window"])
    out["token building (Python)"] = dict(mean_ms=float(np.mean(tb)), p99_ms=float(np.percentile(tb, 99)))
    save("timing.json", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "rules", "timing"]
    for w in which:
        dict(A=eval_stage_a, B=eval_stage_b, C=eval_stage_c, rules=check_rules, timing=timing)[w]()

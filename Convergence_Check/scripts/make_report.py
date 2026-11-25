"""Convergence criteria, raw tables and diagnostic plots from results/ (training logs and results/eval/*.json).
Writes report/raw_tables_auto.md (raw numbers behind the LaTeX report) and report/summary.json."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, OUT = "results", "report"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
SEEDS = [0, 1, 2]
COLORS = ["#0b6e69", "#2f5d8a", "#9a5b00", "#7d3c98", "#555555"]


def read_log(d):
    p = os.path.join(d, "train_log.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def load_json(name):
    p = os.path.join(R, "eval", name)
    return json.load(open(p)) if os.path.exists(p) else None


def smooth(x, w=9):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    y = np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), k, mode="same")
    y[: w // 2] = x[: w // 2]
    y[-(w // 2):] = x[-(w // 2):]
    return y


# ------------------------------------------------------------------ convergence diagnostics
def convergence(stage, suffix=""):
    logs = {s: read_log(f"{R}/stage{stage}_seed{s}{suffix}") for s in SEEDS}
    logs = {s: l for s, l in logs.items() if l is not None}
    if not logs:
        return None
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    panels = [("train_reward", "training reward per request"), ("eval_reward", "validation reward (greedy)"),
              ("entropy", "policy entropy"), ("approx_kl", "approx. KL per update"),
              ("explained_var", "critic explained variance"), ("train_copies", "copies per request")]
    for a, (k, title) in zip(ax.ravel(), panels):
        for c, (s, l) in zip(COLORS, logs.items()):
            x, y = l["iter"], l[k]
            m = ~np.isnan(y)
            if k == "eval_reward":
                a.plot(x[m], y[m], "-o", ms=3, color=c, label=f"seed {s}")
            else:
                a.plot(x[m], y[m], color=c, alpha=0.25, lw=0.8)
                a.plot(x[m], smooth(y[m]), color=c, lw=1.6, label=f"seed {s}")
        if k == "approx_kl":
            a.axhline(0.03, color="#9a5b00", ls="--", lw=1)
        a.set_title(title, fontsize=10)
        a.set_xlabel("iteration", fontsize=8)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Stage {stage}{' (no history)' if suffix else ''}: training diagnostics", fontsize=12)
    fig.tight_layout()
    name = f"conv_stage{stage}{suffix}.png"
    fig.savefig(os.path.join(FIG, name), dpi=130)
    plt.close(fig)
    # criteria
    res = dict(figure=name, seeds={})
    finals = []
    for s, l in logs.items():
        ev = l["eval_reward"]
        m = ~np.isnan(ev)
        evx, evy = l["iter"][m], ev[m]
        n = len(l["iter"])
        tail = l["iter"] >= 0.8 * n
        head = l["iter"] < 0.1 * n
        final = float(np.mean(evy[-3:]))
        first = float(evy[0])
        # premature collapse = entropy near 0 BEFORE the validation reward has reached 95% of its final value
        reached = evx[evy >= 0.95 * final] if final > 0 else evx[-1:]
        t_conv = float(reached[0]) if len(reached) else float(evx[-1])
        pre = l["iter"] < t_conv
        ent_pre = float(np.nanmin(l["entropy"][pre])) if pre.any() else float(l["entropy"][0])
        finals.append(final)
        last_q = evx >= 0.75 * n
        slope = float(np.polyfit(evx[last_q], evy[last_q], 1)[0]) if last_q.sum() >= 3 else np.nan
        res["seeds"][s] = dict(
            eval_first=first, eval_final=final, eval_best=float(np.max(evy)), improvement=final - first,
            slope_last_quarter_per_100it=slope * 100,
            entropy_start=float(np.nanmean(l["entropy"][head])), entropy_end=float(np.nanmean(l["entropy"][tail])),
            entropy_min=float(np.nanmin(l["entropy"])), iter_reward_95pct=t_conv, entropy_min_before_95pct=ent_pre,
            kl_median=float(np.nanmedian(l["approx_kl"])), kl_frac_above_003=float(np.mean(l["approx_kl"] > 0.03)),
            ev_start=float(np.nanmean(l["explained_var"][head])), ev_end=float(np.nanmean(l["explained_var"][tail])),
            copies_end=float(np.nanmean(l["train_copies"][tail])), minutes=float(l["time_s"][-1] / 60))
    finals = np.array(finals)
    res["final_mean"] = float(finals.mean())
    res["final_spread"] = float(finals.max() - finals.min())
    res["final_rel_spread"] = float((finals.max() - finals.min()) / max(abs(finals.mean()), 1e-6))
    return res


def crit_conv(c, ev_floor=0.3):
    """Generic convergence criteria from the documentation (Section 8.3). ev_floor: Stage A uses half of the measured
    noise ceiling of the explained variance (results/eval/critic_ceiling.json), since 0.3 is above that ceiling."""
    s = list(c["seeds"].values())
    return dict(
        learns=all(x["improvement"] > 0 for x in s),
        plateau=all(abs(x["slope_last_quarter_per_100it"]) < 0.05 for x in s),
        seeds_agree=c["final_rel_spread"] <= 0.10 or c["final_spread"] <= 0.03,
        critic_learns=all(x["ev_end"] > max(ev_floor, x["ev_start"]) for x in s),
        entropy_gradual=all(x["entropy_end"] < x["entropy_start"] and x["entropy_min_before_95pct"] > 0.02 for x in s),
        kl_small=all(x["kl_median"] <= 0.03 and x["kl_frac_above_003"] <= 0.2 for x in s),
    )


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def mark(b):
    return "PASS" if b else "FAIL"


def main():
    L = []
    w = L.append
    conv = {st: convergence(st) for st in "ABC"}
    conv["H"] = convergence("H", "_dmax1_aux")
    ceil = load_json("critic_ceiling.json")
    ev_floor = {"A": 0.5 * float(np.mean([v["ceiling"] for v in ceil.values()])) if ceil else 0.3}
    stats = load_json("adaptation_stats.json")
    gen = load_json("generalisation.json")
    simc = load_json("simulator_checks.json")
    conv_nohist = convergence("C", "_nohist")
    A, B, C = load_json("stage_A.json"), load_json("stage_B.json"), load_json("stage_C.json")
    rules, timing = load_json("rules.json"), load_json("timing.json")
    verdict_items = []

    w("# Convergence check: report\n")
    w("Synthetic data only. Everything (simulator, data, model, training, evaluation) is in this folder; "
      "nothing from the measured datasets or other project code is used. The checks follow Section 8 "
      "(\"Step 0: convergence demo on synthetic data\") of the method documentation.\n")

    # ---------------- convergence tables
    w("## 1. Does training converge?\n")
    w("Criteria (documentation Section 8.3): reward rises then flattens; seeds agree (final validation reward within "
      "about ±5%); the critic explains more of the returns over time; entropy falls gradually (no collapse); "
      "the policy change per update (approx. KL) stays small (about 0.01–0.03).\n")
    for st in "ABCH":
        c = conv[st]
        if not c:
            continue
        cr = crit_conv(c, ev_floor.get(st, 0.3))
        w(f"### Stage {st}\n")
        w(f"![Stage {st} diagnostics](figures/{c['figure']})\n")
        w("| Seed | validation reward start → final (best) | slope, last quarter (per 100 it) | entropy start → end | "
          "median KL | explained var. start → end | copies/request (end) | minutes |")
        w("|---|---|---|---|---|---|---|---|")
        for s, x in c["seeds"].items():
            w(f"| {s} | {fmt(x['eval_first'])} → {fmt(x['eval_final'])} ({fmt(x['eval_best'])}) | "
              f"{fmt(x['slope_last_quarter_per_100it'])} | {fmt(x['entropy_start'], 2)} → {fmt(x['entropy_end'], 2)} | "
              f"{fmt(x['kl_median'], 4)} | {fmt(x['ev_start'], 2)} → {fmt(x['ev_end'], 2)} | {fmt(x['copies_end'], 2)} | "
              f"{fmt(x['minutes'], 1)} |")
        w("")
        w(f"Final validation reward across seeds: mean {fmt(c['final_mean'])}, spread {fmt(c['final_spread'])} "
          f"({fmt(100 * c['final_rel_spread'], 1)}% of the mean).\n")
        w("| Criterion | Result |\n|---|---|")
        names = dict(learns="reward improves from the start", plateau="reward flattens (small slope at the end)",
                     seeds_agree="3 seeds agree (spread ≤ 10% of mean or ≤ 0.03)",
                     critic_learns="critic explained variance rises (> 0.3 at the end; Stage A: > half its noise ceiling)",
                     entropy_gradual="entropy falls gradually, no premature collapse", kl_small="policy change per update small")
        for k, v in cr.items():
            w(f"| {names[k]} | {mark(v)} |")
            verdict_items.append((f"Stage {st}: {names[k]}", v))
        w("")

    # ---------------- Stage A
    if A:
        w("## 2. Stage A: one request at a time vs the brute-force best subset\n")
        w("Fixed system with 2 GPU-like and 2 CPU-like servers; one request at a time on idle servers. For every test "
          "request, the expected reward of all 15 server subsets is computed by Monte Carlo (4000 samples, true "
          "latency model); the choice counts as correct if it reaches at least 99% of the best expected reward.\n")
        w("| Policy | decisions | within 1% of best | exact best subset | mean regret | mean best reward |")
        w("|---|---|---|---|---|---|")
        for name, x in list(A["models"].items()) + list(A["baselines"].items()):
            lab = f"Proposed model {name}" if name.startswith("seed") else name
            w(f"| {lab} | {x['n']} | {fmt(100 * x['within_1pct'], 1)}% | {fmt(100 * x['exact'], 1)}% | "
              f"{fmt(x['mean_regret'], 4)} | {fmt(x['mean_best'])} |")
        w("")
        w("Per service (Proposed model, mean over seeds). *2+ copies* = share of requests where two or more servers are used.\n")
        w("| Service | within 1% of best | brute force says 2+ copies | model uses 2+ copies | agree on 1 vs 2+ copies |")
        w("|---|---|---|---|---|")
        names = ["speech-like (GPU outliers 15%)", "detect-like (no GPU outliers)", "segment-like (no GPU outliers)"]
        okA = []
        for s in range(3):
            vals = [m["per_service"][str(s)] for m in A["models"].values()]
            mean = lambda k: float(np.mean([v[k] for v in vals]))
            w(f"| {names[s]} | {fmt(100 * mean('within_1pct'), 1)}% | {fmt(100 * mean('bf_says_2plus'), 1)}% | "
              f"{fmt(100 * mean('model_2plus'), 1)}% | {fmt(100 * mean('agree_on_copies'), 1)}% |")
        w("")
        within = [m["within_1pct"] for m in A["models"].values()]
        passA = all(x >= 0.95 for x in within)
        verdict_items.append(("Stage A: ≥95% of decisions within 1% of the brute-force best (every seed)", passA))
        w(f"**Criterion: ≥ 95% of decisions within 1% of the best, for every seed: {mark(passA)}** "
          f"(seeds: {', '.join(fmt(100 * x, 1) + '%' for x in within)}).\n")

    # ---------------- Stage B
    if B:
        w("## 3. Stage B: batches on a fixed system vs baselines\n")
        w("Fixed system with 5 servers (2 GPU-like, 3 CPU-like), random and bursty arrivals. 12 test episodes of 150 "
          "requests per load, the same for every policy. Latencies are normalised by each request's hard deadline "
          "(1.0 = exactly at the hard deadline). Greedy-Predict is privileged: it knows the true speeds and outlier "
          "rates and uses Monte-Carlo estimates of every option.\n")
        loads = list(B.keys())
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        pols = [p for p in B[loads[0]].keys() if p != "Oracle bound"]
        for a, key, title in zip(ax, ["mean_reward", "p95_norm", "copies_per_req"],
                                 ["mean reward per request (higher is better)", "p95 latency / hard deadline (lower is better)",
                                  "copies per request"]):
            xs = np.arange(len(loads))
            wdt = 0.8 / len(pols)
            for i, p in enumerate(pols):
                a.bar(xs + i * wdt, [B[l][p][key] for l in loads], wdt, label=p,
                      color=(COLORS[i % 3] if p.startswith("Proposed model") else ["#bbbbbb", "#999999", "#777777", "#444444"][(i - 3) % 4]))
            a.set_xticks(xs + 0.4 - wdt / 2)
            a.set_xticklabels(loads, fontsize=8)
            a.set_title(title, fontsize=9)
            a.grid(axis="y", alpha=0.3)
        ax[0].legend(fontsize=6.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "stageB_comparison.png"), dpi=130)
        plt.close(fig)
        w("![Stage B comparison](figures/stageB_comparison.png)\n")
        okB_beat, okB_close, gaps = [], [], []
        for l in loads:
            w(f"**Load: {l}**\n")
            w("| Policy | mean reward | mean P(T) | hard-deadline misses | p50 | p95 | p99 | copies/request | requests per decision |")
            w("|---|---|---|---|---|---|---|---|---|")
            for p, x in B[l].items():
                if p == "Oracle bound":
                    w(f"| {p} (latency only) | – | – | – | {fmt(x['p50_norm'], 2)} | {fmt(x['p95_norm'], 2)} | "
                      f"{fmt(x['p99_norm'], 2)} | – | – |")
                else:
                    w(f"| {p} | {fmt(x['mean_reward'])} | {fmt(x['mean_P'])} | {fmt(100 * x['miss_hard'], 1)}% | "
                      f"{fmt(x['p50_norm'], 2)} | {fmt(x['p95_norm'], 2)} | {fmt(x['p99_norm'], 2)} | "
                      f"{fmt(x['copies_per_req'], 2)} | {fmt(x['mean_c'], 2)} |")
            w("")
            tf = [x["mean_reward"] for p, x in B[l].items() if p.startswith("Proposed model")]
            simple = max(B[l][p]["mean_reward"] for p in ["Random-1", "Redundancy-2", "JSQ"])
            gp = B[l]["Greedy-Predict (privileged)"]["mean_reward"]
            okB_beat.append(min(tf) > simple)
            gaps.append(gp - float(np.mean(tf)))
        passB1 = all(okB_beat)
        verdict_items.append(("Stage B: every seed beats Random, Redundancy-2 and JSQ at every load", passB1))
        w(f"**Criterion: every proposed-model seed beats Random-1, Redundancy-2 and JSQ at every load: {mark(passB1)}**\n")
        w("Gap to the privileged Greedy-Predict (its reward minus the proposed model's mean; negative = Proposed model better): "
          + ", ".join(f"{l} {fmt(g)}" for l, g in zip(loads, gaps)) + ".\n")
        cps = [np.mean([B[l][p]["copies_per_req"] for p in B[l] if p.startswith("Proposed model")]) for l in ["low", "medium", "high"] if l in B]
        passB2 = len(cps) == 3 and cps[0] > cps[2]
        verdict_items.append(("Stage B: fewer copies at high load than at low load (sanity scenario 3)", passB2))
        w(f"**Sanity scenario 3 (high load → fewer copies): copies per request low/medium/high = "
          f"{', '.join(fmt(c, 2) for c in cps)}: {mark(passB2)}**\n")

    # ---------------- Stage C
    if C:
        w("## 4. Stage C: meta-RL across random systems, adaptation from history\n")
        w("Every training episode is a new random system (3–6 servers, 1–2 GPU-like, random speeds, capacities, outlier "
          "rates, loads, users); server speeds are never shown to the model, and in half of the systems one GPU-like "
          "server is secretly 3× slower than it looks. Test: 48 unseen random systems. The same model trained without "
          "history tokens is the ablation.\n")
        w("| Policy | mean reward | mean P(T) | hard-deadline misses | p95 | p99 | copies/request |")
        w("|---|---|---|---|---|---|---|")
        for p, x in C["validation"].items():
            w(f"| {p} | {fmt(x['mean_reward'])} | {fmt(x['mean_P'])} | {fmt(100 * x['miss_hard'], 1)}% | "
              f"{fmt(x['p95_norm'], 2)} | {fmt(x['p99_norm'], 2)} | {fmt(x['copies_per_req'], 2)} |")
        w("")
        hist = [x["mean_reward"] for p, x in C["validation"].items() if p.startswith("Proposed model (history)")]
        nohist = [x["mean_reward"] for p, x in C["validation"].items() if p.startswith("Proposed model (no history)")]
        w(f"Open loop (this table): history {', '.join(fmt(x) for x in hist)}; no history {', '.join(fmt(x) for x in nohist)} "
          "- not discriminative: the running-copy tokens (elapsed time per server) already reveal a slow server.\n")
        if stats:
            g = stats["C_closed_loop"]
            passC1 = all(v["history_gain"]["above_zero"] for v in g.values())
            verdict_items.append(("Stage C models, closed loop: history gain > 0 (95% CI, every seed)", passC1))
            w(f"**Closed loop, Stage C models: reward gain from history (95% CI): "
              + ", ".join(f"{k} {fmt(v['history_gain']['mean'])} [{fmt(v['history_gain']['lo'])}, {fmt(v['history_gain']['hi'])}]"
                          for k, v in g.items()) + f": {mark(passC1)}**\n")
        ad = C["adaptation"]
        buckets = C["adaptation_buckets"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (p, x) in enumerate(ad.items()):
            style = "-o" if "history)" in p and "no" not in p else ("--s" if "no history" in p else ":^")
            col = COLORS[0] if "(history)" in p else (COLORS[2] if "no history" in p else "#777777")
            ax.plot(range(len(buckets)), x["slowed"], style, color=col, ms=4, alpha=0.9,
                    label=p if p.endswith("seed0") or not p.startswith("Proposed model") else None)
        ref = np.nanmean([x["normal"] for p, x in ad.items() if "(history)" in p], axis=0)
        ax.plot(range(len(buckets)), ref, color="black", lw=1, ls="-.", label="same server when NOT slowed (history model)")
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(buckets)
        ax.set_xlabel("decisions since the start of the episode")
        ax.set_ylabel("share of copies sent to the slowed server")
        ax.set_title("Adaptation: one GPU-like server secretly 3x slower", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "stageC_adaptation.png"), dpi=130)
        plt.close(fig)
        w("![Stage C adaptation](figures/stageC_adaptation.png)\n")
        w("| Policy | share of copies to the slowed server, by decision index (" + ", ".join(buckets) + ") | "
          "same server when not slowed | reward slowed / normal |")
        w("|---|---|---|---|")
        for p, x in ad.items():
            w(f"| {p} | {' → '.join(fmt(v, 2) for v in x['slowed'])} | {' → '.join(fmt(v, 2) for v in x['normal'])} | "
              f"{fmt(x['reward_slowed'])} / {fmt(x['reward_normal'])} |")
        w("")
        hs = [x for p, x in ad.items() if "(history)" in p]
        passC2 = bool(stats) and all(v["avoidance_gap_hist"]["above_zero"] for v in stats["C_open_loop"].values())
        verdict_items.append(("Stage C: the secretly slowed server is avoided as results arrive (sanity scenario 4)", passC2))
        w(f"**Sanity scenario 4 (secretly slowed server avoided after results arrive): {mark(passC2)}** - paired over twin "
          "systems: (share on the unslowed twin - share on the slowed twin), late minus early in the episode; 95% CI above 0 "
          "for every seed.\n")

    # ---------------- history test (only history can reveal speeds)
    H = load_json("history_test.json")
    convH = {"history": convergence("H", "_dmax1_aux"), "no history": convergence("H", "_nohist_dmax1_aux")}
    if H:
        w("## 4b. History test: only the history tokens can reveal which server is slow\n")
        w("Closed loop (one request at a time, every server idle at each decision), one copy per request (d_max = 1), "
          f"random systems with hidden speeds; {H['n_systems']} unseen systems, each once with one GPU-like server "
          "secretly 3x slower and once without. A model without history sees the same state at every decision.\n")
        runs_h = {k: v for k, v in H["runs"].items() if k.startswith("stageH") and "nohist" not in k}
        runs_n = {k: v for k, v in H["runs"].items() if "nohist" in k}
        others = {k: v for k, v in H["runs"].items() if not k.startswith("stageH")}
        w("| Policy | reward slowed / normal | reward by request index, slowed (" + ", ".join(H["buckets"]) +
          ") | share to slowed server | same server, not slowed |")
        w("|---|---|---|---|---|")
        for k, x in list(runs_h.items()) + list(runs_n.items()) + list(others.items()):
            w(f"| {k} | {fmt(x['reward_slowed'])} / {fmt(x['reward_normal'])} | "
              f"{' → '.join(fmt(v) for v in x['reward_curve_slowed'])} | {' → '.join(fmt(v, 2) for v in x['share_slowed'])} | "
              f"{' → '.join(fmt(v, 2) for v in x['share_normal'])} |")
        w("")
        mh = [(x["reward_slowed"] + x["reward_normal"]) / 2 for x in runs_h.values()]
        mn = [(x["reward_slowed"] + x["reward_normal"]) / 2 for x in runs_n.values()]
        passH1 = bool(stats) and all(v["history_gain"]["above_zero"] for v in stats["H"].values())
        gap = lambda x, i: x["share_normal"][i] - x["share_slowed"][i]
        passH2 = bool(stats) and all(v["avoidance_gap_hist"]["above_zero"] for v in stats["H"].values())
        verdict_items.append(("History test: reward gain from history > 0 (paired, 95% CI, every seed)", passH1))
        verdict_items.append(("History test: slowed server avoided after the first results (paired twins, 95% CI, every seed)", passH2))
        w(f"**History beats no history (every seed pair): {mark(passH1)}** (history {', '.join(fmt(x) for x in mh)}; "
          f"no history {', '.join(fmt(x) for x in mn)}).  \n**Slowed server avoided as results arrive: {mark(passH2)}**\n")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        xs = range(len(H["buckets"]))
        for grp, col, sty in [(runs_h, COLORS[0], "-o"), (runs_n, COLORS[2], "--s")]:
            for i, (k, x) in enumerate(grp.items()):
                lab = ("with history" if grp is runs_h else "without history") if i == 0 else None
                ax[0].plot(xs, x["share_slowed"], sty, color=col, ms=4, label=lab)
                ax[1].plot(xs, x["reward_curve_slowed"], sty, color=col, ms=4, label=lab)
        gp = others.get("Greedy-Predict (privileged)")
        if gp:
            ax[0].plot(xs, gp["share_slowed"], ":^", color="#777777", label="Greedy-Predict (privileged)")
            ax[1].plot(xs, gp["reward_curve_slowed"], ":^", color="#777777", label="Greedy-Predict (privileged)")
        ax[0].set_title("share of requests sent to the secretly slowed server", fontsize=10)
        ax[1].set_title("mean reward on the slowed systems", fontsize=10)
        for a in ax:
            a.set_xticks(list(xs)); a.set_xticklabels(H["buckets"]); a.set_xlabel("request index in the episode")
            a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "history_test.png"), dpi=130)
        plt.close(fig)
        w("![History test](figures/history_test.png)\n")

    # ---------------- rules and timing
    if rules:
        w("## 5. Hard rules (masks)\n")
        w("| Rule | Result | Details |\n|---|---|---|")
        lab = dict(urgent_request_always_sent="an urgent request is always sent (sanity scenario 5)",
                   idle_server_forces_one_edge="an idle server with waiting requests forces at least one edge",
                   holding_back_allowed="otherwise holding requests back is possible",
                   capacity_copy_limits="capacity, copy limit and no repeated edge are never violated")
        for k, v in rules.items():
            det = ", ".join(f"{a}={b}" for a, b in v.items() if a != "passed")
            w(f"| {lab[k]} | {mark(v['passed'])} | {det} |")
            verdict_items.append((f"Rule: {lab[k]}", v["passed"]))
        w("")
    if timing:
        w("## 6. Decision time (one decision, CPU, 1 thread, PyTorch eager mode)\n")
        w("| Part | mean (ms) | p50 (ms) | p99 (ms) |\n|---|---|---|---|")
        for k, v in timing.items():
            w(f"| {k} | {fmt(v.get('mean_ms'), 2)} | {fmt(v.get('p50_ms'), 2)} | {fmt(v.get('p99_ms'), 2)} |")
        w("")

    if simc:
        verdict_items.insert(0, (f"Simulator unit checks ({sum(v['passed'] for v in simc.values())} of {len(simc)})",
                                 all(v["passed"] for v in simc.values())))
    if gen:
        conds = [k for k in gen if isinstance(gen[k], dict) and "Redundancy-2" in gen[k]]
        okg = []
        for k in conds:
            tf = [v["mean_reward"] for p, v in gen[k].items() if p.startswith("stageC_seed") and "nohist" not in p]
            okg.append(min(tf) > max(v["mean_reward"] for p, v in gen[k].items() if not p.startswith("stageC")))
        verdict_items.append((f"Generalisation: every seed beats every baseline in all {len(conds)} conditions", all(okg)))
    # ---------------- verdict
    w("## 7. Summary of checks\n")
    w("| Check | Result |\n|---|---|")
    for k, v in verdict_items:
        w(f"| {k} | {mark(v)} |")
    w("")
    n_pass = sum(1 for _, v in verdict_items if v)
    w(f"**{n_pass} of {len(verdict_items)} checks passed.**\n")
    with open(os.path.join(OUT, "raw_tables_auto.md"), "w") as f:
        f.write("\n".join(L))
    json.dump(dict(conv=conv, conv_nohist=conv_nohist, conv_H=convH, checks=verdict_items), open(os.path.join(OUT, "summary.json"), "w"),
              indent=1, default=str)
    print("\n".join(L[-len(verdict_items) - 6:]))


if __name__ == "__main__":
    main()

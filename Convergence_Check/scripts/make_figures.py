"""Figures of the LaTeX report (report/figures/*.pdf) from results/.
Usage: ./run.sh scripts/make_figures.py"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, FIG = "results", "report/figures"
os.makedirs(FIG, exist_ok=True)
# categorical slots, validated (colour-blind separation and normal-vision floor pass on a light surface)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 7, "xtick.major.size": 0,
    "ytick.major.size": 0, "pdf.fonttype": 42,
})


def read_log(d):
    p = os.path.join(R, d, "train_log.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    out = {}
    for k in rows[0]:
        v = []
        for r in rows:
            try:
                v.append(float(r[k]))
            except (TypeError, ValueError):
                v.append(np.nan)
        out[k] = np.array(v)
    return out


def js(name):
    p = os.path.join(R, "eval", name)
    return json.load(open(p)) if os.path.exists(p) else None


EPISODES_PER_ITER = 16          # every PPO iteration collects 16 complete training episodes


def fig_training():
    """Convergence against the number of training episodes (top axis: PPO iterations, 1 iteration = 16 episodes).
    Row 1: training reward (mean reward per request of each batch of 16 episodes, faint; moving average over 10
    batches = 160 episodes, solid). Row 2: reward on the fixed validation set, measured every 160 episodes."""
    from matplotlib.ticker import FuncFormatter
    panels = [("A  single requests", "stageA_seed{}"), ("B  fixed system", "stageB_seed{}"),
              ("C  random systems", "stageC_seed{}"), ("H  history test", "stageH_seed{}_dmax1_aux")]
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.75), sharex="col")
    kfmt = FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k")
    for c, (title, pat) in enumerate(panels):
        top, bot = ax[0, c], ax[1, c]
        n_ep = 0
        for s in range(3):
            l = read_log(pat.format(s))
            if l is None:
                continue
            ep = (l["iter"] + 1) * EPISODES_PER_ITER          # episodes collected so far
            n_ep = max(n_ep, ep[-1])
            tr = l["train_reward"]
            top.plot(ep, tr, color=BLUE, lw=0.5, alpha=0.18)
            w = 10
            if len(tr) > w:
                top.plot(ep[w - 1:], np.convolve(tr, np.ones(w) / w, mode="valid"), color=BLUE, lw=1.2, alpha=0.9)
            m = ~np.isnan(l["eval_reward"])
            bot.plot(ep[m], l["eval_reward"][m], color=BLUE, lw=1.2, alpha=0.9, solid_capstyle="round")
        if c in (1, 2):                                        # curriculum: wider load range after the first quarter
            for a in (top, bot):
                a.axvline(0.25 * n_ep, color=MUTED, lw=0.7, ls=(0, (2, 2)))
            top.text(0.25 * n_ep, 1.0, " harder loads", transform=top.get_xaxis_transform(), fontsize=6,
                     color=MUTED, va="top", ha="left")
        top.set_title(title)
        sec = top.secondary_xaxis("top", functions=(lambda e: e / EPISODES_PER_ITER, lambda i: i * EPISODES_PER_ITER))
        sec.set_xlabel("PPO iteration", fontsize=6.5, color=MUTED)
        sec.tick_params(labelsize=6.5, colors=MUTED, length=0)
        bot.set_xlabel("training episodes")
        bot.xaxis.set_major_formatter(kfmt)
    ax[0, 0].set_ylabel("training reward\n(per 16-episode batch)")
    ax[1, 0].set_ylabel("validation reward\n(fixed episodes)")
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_training.pdf"))
    plt.close(fig)


def fig_stage_b():
    B = js("stage_B.json")
    if not B:
        return
    loads = list(B.keys())
    pols = [("Proposed model (mean of 3 seeds, bar = range)", None, BLUE),
            ("Greedy-Predict (knows true speeds)", "Greedy-Predict (privileged)", ORANGE),
            ("Redundancy-2", "Redundancy-2", AQUA), ("JSQ", "JSQ", YELLOW)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.45))
    for a, key, title in [(ax[0], "mean_reward", "Mean reward per request  (higher is better)"),
                          (ax[1], "p95_norm", "p95 latency / hard deadline  (lower is better)")]:
        x = np.arange(len(loads))
        w = 0.19
        for i, (lab, name, col) in enumerate(pols):
            if name is None:
                vals = np.array([[v[key] for p, v in B[l].items() if p.startswith("Proposed model")] for l in loads])
                a.bar(x + (i - 1.5) * w, vals.mean(1), w - 0.03, color=col, label=lab)
                a.vlines(x + (i - 1.5) * w, vals.min(1), vals.max(1), color=INK, lw=0.9)
            else:
                a.bar(x + (i - 1.5) * w, [B[l][name][key] for l in loads], w - 0.03, color=col, label=lab)
        a.set_xticks(x)
        a.set_xticklabels([f"{l} load" if l != "bursty" else "bursty" for l in loads])
        a.set_title(title)
        a.grid(axis="x", visible=False)
    ax[1].axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 2)), label="hard deadline")
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    hd = [(h, l) for h, l in zip(h1, l1) if l == "hard deadline"]
    fig.legend(h0 + [x[0] for x in hd], l0 + [x[1] for x in hd], loc="lower center", ncol=5, handlelength=1.2, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_stageB.pdf"))
    plt.close(fig)


def fig_history():
    H = js("history_test.json")
    if not H:
        return
    runs = H["runs"]
    hist = [v for k, v in runs.items() if k.startswith("stageH") and "nohist" not in k]
    noh = [v for k, v in runs.items() if "nohist" in k]
    gp = runs.get("Greedy-Predict (privileged)")
    x = np.arange(len(H["buckets"]))

    def lab(b):
        lo, hi = (int(v) + 1 for v in b.split("-"))
        return "1st" if lo == hi else f"{lo}–{hi}"
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.35))
    for a, key, title in [(ax[0], "share_slowed", "Share of requests sent to the secretly slowed server"),
                          (ax[1], "reward_curve_slowed", "Mean reward on the slowed systems")]:
        for grp, col, name in [(hist, BLUE, "with history"), (noh, ORANGE, "without history")]:
            arr = np.array([g[key] for g in grp])
            for row in arr:
                a.plot(x, row, color=col, lw=0.7, alpha=0.35)
            a.plot(x, arr.mean(0), "-o", color=col, lw=2, ms=4.5, mec="white", mew=1.2, label=f"Proposed model {name}")
        if gp:
            a.plot(x, gp[key], "-o", color=AQUA, lw=2, ms=4.5, mec="white", mew=1.2, label="Greedy-Predict (knows true speeds)")
        a.set_xticks(x)
        a.set_xticklabels([lab(b) for b in H["buckets"]])
        a.set_xlabel("request number in the episode")
        a.set_title(title)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, handlelength=1.8, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_history.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_training()
    fig_stage_b()
    fig_history()
    print("figures written to", FIG)

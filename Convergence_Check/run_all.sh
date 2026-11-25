#!/usr/bin/env bash
# Reproduce the whole convergence check with the final pipeline: data, training, evaluation, report (LaTeX PDF).
# Runs inside this folder's virtual environments only (see run.sh): .venv (CPU) and .venv_gpu (CUDA, PPO update step).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
fi
cap() { systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 "$@"; }
./run.sh scripts/make_data.py
scripts/final_runs.sh
cap ./run.sh scripts/evaluate.py A B
scripts/run_evals_final.sh                       # simulator checks, rules, timing, C, history tests, generalisation
cap ./run.sh scripts/check_critic_ceiling.py
cap ./run.sh scripts/eval_adaptation_stats.py
cap ./run.sh scripts/eval_ckpt_plateau.py
./run.sh scripts/make_report.py                  # criteria -> report/summary.json, raw_tables_auto.md
./run.sh scripts/make_figures.py
(cd report && ~/.local/bin/tectonic Convergence_Check_Report.tex)

#!/usr/bin/env bash
# All evaluations of the convergence check (final checkpoints), then the figures and tables.
cd "$(dirname "$0")/.."
set -e
./run.sh scripts/check_simulator.py
./run.sh scripts/evaluate.py rules timing A B C
HIST_DMAX=1 ./run.sh scripts/eval_history.py results/stageH_seed{0,1,2}_dmax1_aux results/stageH_seed{0,1,2}_nohist_dmax1_aux
./run.sh scripts/eval_generalisation.py
./run.sh scripts/make_report.py

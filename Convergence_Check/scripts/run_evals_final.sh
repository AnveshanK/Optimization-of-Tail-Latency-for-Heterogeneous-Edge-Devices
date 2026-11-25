#!/usr/bin/env bash
# Final evaluations (final checkpoints of the Delta = 5 ms pipeline), one at a time, each capped at 4 GB of RAM.
cd "$(dirname "$0")/.."
cap() { systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 "$@"; }
date '+%H:%M start'
cap ./run.sh scripts/check_simulator.py | tail -1
cap ./run.sh scripts/evaluate.py rules timing C | tail -4
HIST_DMAX=1 cap ./run.sh scripts/eval_history.py results/stageH_seed{0,1,2}_dmax1_aux results/stageH_seed{0,1,2}_nohist_dmax1_aux | grep -v "^Random\|^JSQ"
HIST_OUT=results/eval/history_test_stageC.json cap ./run.sh scripts/eval_history.py results/stageC_seed{0,1,2} results/stageC_seed{0,1,2}_nohist | grep "^stage\|Greedy"
cap ./run.sh scripts/eval_generalisation.py | tail -30
date '+%H:%M evals done'

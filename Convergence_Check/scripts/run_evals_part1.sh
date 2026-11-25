#!/usr/bin/env bash
# Evaluations that do not need Stage B, one at a time, each capped at 4 GB of RAM.
cd "$(dirname "$0")/.."
cap() { systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 "$@"; }
date '+%H:%M start'
cap ./run.sh scripts/check_simulator.py | tail -1
cap ./run.sh scripts/evaluate.py rules C | tail -2
HIST_DMAX=1 cap ./run.sh scripts/eval_history.py results/stageH_seed{0,1,2}_dmax1_aux results/stageH_seed{0,1,2}_nohist_dmax1_aux | grep -v "^Random\|^JSQ"
HIST_DMAX=1 HIST_OUT=results/eval/history_test_noaux.json cap ./run.sh scripts/eval_history.py results/stageH_seed{0,1,2}_dmax1 results/stageH_seed{0,1,2}_nohist_dmax1 results/stageH_seed0_dmax1_long | grep "^stageH"
HIST_OUT=results/eval/history_test_stageC.json cap ./run.sh scripts/eval_history.py results/stageC_seed{0,1,2} results/stageC_seed{0,1,2}_nohist | grep "^stage\|Greedy"
cap ./run.sh scripts/eval_generalisation.py
date '+%H:%M part 1 done'

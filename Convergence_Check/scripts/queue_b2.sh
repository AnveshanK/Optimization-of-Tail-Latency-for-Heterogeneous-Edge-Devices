#!/usr/bin/env bash
# Stage B rerun with the final pipeline (same settings as Stage C: Delta = 5 ms, aux 0.5, 400 iterations),
# 32-episode validation and a checkpoint every 25 iterations (to measure the plateau on the full test set).
cd "$(dirname "$0")/.."
T="--threads 1 --lr-final 1e-4 --aux 0.5 --delta 5 --ckpt-every 25"
for s in 0 1 2; do
  echo "echo 800 > /proc/self/oom_score_adj; CONVCHECK_VENV=.venv_gpu ./run.sh scripts/train_stage.py B $s $T --device cuda --tag _v2 > logs/B_${s}_v2.out 2>&1"
done | xargs -P 3 -I{} bash -c "{}"
date '+%H:%M queue done'

#!/usr/bin/env bash
# Final-pipeline retraining with Delta = 5 ms (C and H), at most 4 training processes at a time (RAM safety).
cd "$(dirname "$0")/.."
T="--threads 1 --lr-final 1e-4 --aux 0.5 --delta 5"
{
for s in 0 1 2; do
  echo "CONVCHECK_VENV=.venv_gpu ./run.sh scripts/train_stage.py C $s $T --device cuda --tag _d5 > logs/C_${s}_d5.out 2>&1"
  echo "CONVCHECK_VENV=.venv_gpu ./run.sh scripts/train_stage.py C $s $T --device cuda --no-history --tag _d5 > logs/Cn_${s}_d5.out 2>&1"
done
for s in 0 1 2; do
  echo "./run.sh scripts/train_stage.py H $s $T --dmax 1 --tag _dmax1_aux_d5 > logs/H_${s}_d5.out 2>&1"
  echo "./run.sh scripts/train_stage.py H $s $T --dmax 1 --no-history --tag _dmax1_aux_d5 > logs/Hn_${s}_d5.out 2>&1"
done
} | xargs -P 4 -I{} bash -c "{}"
date '+%H:%M queue done'

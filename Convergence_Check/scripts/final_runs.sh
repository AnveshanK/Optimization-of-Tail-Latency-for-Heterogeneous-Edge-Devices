#!/usr/bin/env bash
# Final training runs of the convergence check (final pipeline, 3 seeds each), at most 4 processes at a time (RAM safety).
# Every run: lr 1e-3 decayed linearly to 1e-4, STOP bias 0.5 (default), final checkpoint used.
#  A : 300 iterations, Delta -> 0 (requests are independent in Stage A: credit = the request's own reward)
#  B : 400 iterations, Delta = 5 ms, auxiliary speed loss 0.5, checkpoints every 25 iterations (GPU update step)
#  C : 400 iterations, Delta = 5 ms, auxiliary speed loss 0.5, with / without history (GPU update step)
#  H : 200 iterations, history test (closed loop, d_max = 1), Delta = 5 ms, aux 0.5, with / without history
# Diagnostics that are not part of the final pipeline are kept in results/diagnostics/.
cd "$(dirname "$0")/.."
mkdir -p logs
T="--threads 1 --lr-final 1e-4"
G="CONVCHECK_VENV=.venv_gpu"
{
for s in 0 1 2; do
  echo "./run.sh scripts/train_stage.py A $s $T --delta 0.001 > logs/A_$s.out 2>&1"
  echo "$G ./run.sh scripts/train_stage.py B $s $T --aux 0.5 --delta 5 --ckpt-every 25 --device cuda > logs/B_$s.out 2>&1"
  echo "$G ./run.sh scripts/train_stage.py C $s $T --aux 0.5 --delta 5 --device cuda > logs/C_$s.out 2>&1"
  echo "$G ./run.sh scripts/train_stage.py C $s $T --aux 0.5 --delta 5 --device cuda --no-history > logs/Cn_$s.out 2>&1"
  echo "./run.sh scripts/train_stage.py H $s $T --aux 0.5 --delta 5 --dmax 1 --tag _dmax1_aux > logs/H_$s.out 2>&1"
  echo "./run.sh scripts/train_stage.py H $s $T --aux 0.5 --delta 5 --dmax 1 --no-history --tag _dmax1_aux > logs/Hn_$s.out 2>&1"
done
} | xargs -P 4 -I{} bash -c "echo 800 > /proc/self/oom_score_adj; {}"
echo "all training done"

#!/usr/bin/env bash
# Memory watchdog for long runs on a 16 GB laptop.
#  - every training/evaluation process of this folder gets oom_score_adj 800, so if memory ever runs out the kernel
#    stops one of them instead of the desktop / VS Code
#  - if available RAM drops below LOW_MB, training processes are paused (SIGSTOP) until it is back above HIGH_MB
LOW_MB=${LOW_MB:-2000}; HIGH_MB=${HIGH_MB:-3500}
paused=0
while true; do
  for p in $(pgrep -f "Convergence_Check.*(train_stage|evaluate|eval_|check_simulator|make_)" ; pgrep -f "scripts/(train_stage|evaluate|eval_history|eval_generalisation|eval_a_runs|check_simulator)\.py"); do
    [ -w /proc/$p/oom_score_adj ] && [ "$(cat /proc/$p/oom_score_adj 2>/dev/null)" != "800" ] && echo 800 > /proc/$p/oom_score_adj 2>/dev/null
  done
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$paused" = 0 ] && [ "$avail" -lt "$LOW_MB" ]; then
    pkill -STOP -f "scripts/train_stage.py"; paused=1; echo "$(date +%H:%M:%S) paused training, available ${avail} MB"
  elif [ "$paused" = 1 ] && [ "$avail" -gt "$HIGH_MB" ]; then
    pkill -CONT -f "scripts/train_stage.py"; paused=0; echo "$(date +%H:%M:%S) resumed training, available ${avail} MB"
  fi
  sleep 5
done

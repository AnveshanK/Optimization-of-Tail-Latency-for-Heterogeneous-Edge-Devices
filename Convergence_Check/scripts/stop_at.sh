#!/usr/bin/env bash
# Stop all training runs at a given time (HH:MM) and finalize them.
cd "$(dirname "$0")/.."
until [ "$(date +%H%M)" -ge "$1" ]; do
  ps -eo args | grep -q "[.]venv/bin/python scripts/train_stage" || break
  sleep 30
done
for p in $(ps -eo pid,args | grep "[.]venv/bin/python scripts/train_stage" | awk '{print $1}'); do kill $p; done
sleep 5
./run.sh scripts/finalize_stopped.py

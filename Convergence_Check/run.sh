#!/usr/bin/env bash
# Run a script with this folder's virtual environment only (no system or ROS Python paths).
# CONVCHECK_VENV=.venv_gpu ./run.sh ...   uses the venv with the CUDA build of PyTorch (for --device cuda).
cd "$(dirname "$0")"
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONPATH="$(pwd)" "${CONVCHECK_VENV:-.venv}/bin/python" "$@"

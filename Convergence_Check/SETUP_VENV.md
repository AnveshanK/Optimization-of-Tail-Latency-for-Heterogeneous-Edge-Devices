# Setting up the virtual environments

```bash
cd Convergence_Check
# CPU environment (.venv, ~1.1 GB): rollouts, evaluation, report
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# GPU environment (.venv_gpu, ~5.5 GB): only needed for the PPO update step of Stages B and C (--device cuda)
python3 -m venv .venv_gpu
.venv_gpu/bin/pip install numpy==2.1.3 matplotlib==3.11.2 torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple
```
Versions used for the reported results: torch 2.15.0 (CPU) and 2.15.0+cu132 (GPU), numpy 2.1.3, matplotlib 3.11.2.
Without .venv_gpu everything still runs on the CPU (drop `--device cuda`), only slower.

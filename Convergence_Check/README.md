# Convergence Check on Synthetic Data

Before building the emulator from measured data, this check tests on **synthetic data** that the proposed scheduler
(trained with meta-RL and PPO) can learn at all, and learns the right behaviour. It is described in Chapter 8 of the
BTP report; the full results are in `report/Convergence_Check_Report.pdf`.

Everything here is self-contained: the simulator, the synthetic data, the model, training, evaluation and the report.
No code or data from the rest of the project folder is used.

## Folder

| Path | Contents |
|---|---|
| `convcheck/world.py` | Synthetic edge world: services, servers, users, event-driven simulator (copies, cancellation, progress rule for shared servers, outliers, network time), reward |
| `convcheck/obs.py` | Builds the model input: request, server, running-copy, history and global tokens, edge features, masks |
| `convcheck/model.py` | Scheduler model: encoder, edge-by-edge decoder with the mixed score and STOP, masks, critic |
| `convcheck/ppo.py` | Batched rollouts, time-discounted returns with GAE, PPO update, training loop, logging |
| `convcheck/systems.py`, `convcheck/stages.py` | Synthetic systems and test sets for Stages A, B, C and the history test H |
| `convcheck/baselines.py` | Random, Redundancy-2, join-shortest-queue, privileged Greedy-Predict, Stage A brute force |
| `scripts/` | `train_stage.py`, `final_runs.sh` (training); `evaluate.py`, `eval_history.py`, `eval_generalisation.py`, `eval_adaptation_stats.py`, `eval_ckpt_plateau.py`, `check_simulator.py`, `check_critic_ceiling.py` (evaluation); `make_report.py`, `make_figures.py` (report) |
| `data/` | The synthetic data as JSON (services, systems, test episodes) |
| `results/` | Created when the scripts are run (not stored in the repository, to keep it small): final runs (`stage{A,B,C}_seed*`, `stageH_seed*_dmax1_aux`), `eval/` (JSON results), `diagnostics/` (earlier and ablation runs) |
| `report/` | **`Convergence_Check_Report.pdf`** (final report with verdict) and its `.tex`, figures, `summary.json` and `raw_tables_auto.md` (raw numbers) |

## Run

Everything runs inside this folder's own virtual environment (`.venv`, see `SETUP_VENV.md`); `run.sh` clears any system Python paths.

```bash
./run_all.sh                                   # everything with the final pipeline (a few hours; CPU + laptop GPU)
./run.sh scripts/train_stage.py B 0            # one stage, one seed
./run.sh scripts/evaluate.py B                 # evaluate one stage
./run.sh scripts/make_report.py && ./run.sh scripts/make_figures.py   # criteria and figures
(cd report && ~/.local/bin/tectonic Convergence_Check_Report.tex)  # the PDF report
```

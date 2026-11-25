# Computation-Latency Data Collection Tool

This folder measures how long three edge services take on **this machine** when 1–5 requests run **at the same time**: speech-to-text (`s`), object detection (`d`) and instance segmentation (`p`).
- Each request processes a whole input folder (566 images or 79 audio clips) in its own process, with its own model, like independent requests on an edge server.
- In CPU mode the machine's CPU threads are shared out evenly between the requests running together.
- The data is used to model computation delay on heterogeneous edge servers in this project.

**This file is the quick start.** For the full explanation (design, measurement, every output column, limitations) see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## 1. What you need

- **Operating system:** Linux (Ubuntu recommended), Windows or macOS.
- **Python 3.9 – 3.13** (3.12 recommended) with `venv` (on Ubuntu: `sudo apt install python3-venv`). No administrator rights are needed otherwise.
- **Free disk:** about **6 GB** for CPU mode or **10 GB** for GPU mode (mostly removed again at the end), and an internet connection for the first setup. The script checks this before installing.
- **GPU mode only:** an NVIDIA GPU with its driver installed (`nvidia-smi` must work). Any recent NVIDIA GPU works, e.g. RTX 30 / 40 / 50 series or data-centre cards: the script picks the PyTorch build that works on it. CUDA itself does not need to be installed.

## 2. Before you start: make the machine idle

The measurements are only valid if nothing else is running.
- **Close** browsers, IDEs, chat apps, games, downloads, and other jobs or scripts.
- **Keep the machine plugged in and switched on** for the whole collection.

The script checks this:
- It lists any program that is using the CPU or GPU and asks you to close it.
- It checks again before and during every run. A run disturbed by another program, or measured on battery, is thrown away and repeated automatically.
- It **stops the machine from sleeping** while it runs (on Linux, even when the laptop lid is closed). If the machine sleeps anyway, the affected run is redone.
- **Do not shut the machine down.** If that happens, run the same command again to resume.

## 3. Run it

| System | Command (inside the `Dataset_Collection_Tool` folder) |
|---|---|
| Linux / macOS | `./run.sh` or `python3 run.py` |
| Windows | `run.bat` or `py run.py` |

It first shows what it found on the machine (CPU, GPU, driver, Python, free disk), then asks two questions:

1. **cpu or gpu.** It suggests `gpu` if an NVIDIA GPU was found. Use the mode you were asked to collect.
2. **machine id.** A short unique name, e.g. `lab_i7_11700`. Please use the name you were given.

If nobody answers within 10 minutes, the suggested answer is used, so the script never hangs.

The first time, it installs everything into a private folder (5–15 min): the PyTorch build that suits this machine, the other packages and the speech model (~1.3 GB). Then it starts collecting.

Non-interactive form:

```bash
python3 run.py --mode gpu --machine-id lab_rtx4080
```

**Do the quick check first (2–5 minutes):**

```bash
python3 run.py --mode gpu --machine-id <name> --smoke
```

All 5 runs must end with `ok`, and the last lines must say the data is `consistent`. The output goes to a separate `_smoke` folder.

### How long it takes

| Mode | Full collection (275 runs) on a recent laptop (8 cores / 12 threads) |
|---|---|
| GPU | about 1.5–2 hours |
| CPU only | about 10 hours (up to about twice as long on older or smaller CPUs) |

You can stop at any time with `Ctrl+C`. Running the **same command** again resumes where it left off.

## 4. When it finishes

The script does everything in one go: **setup → data collection → check → cleanup → zip → end**. You don't need to answer anything after the two questions at the start.

1. **The data is checked:** every run must be complete, with one row per request and per file. The result is shown at the end.
2. **Cleanup happens automatically.** The script lists what it installed or downloaded, with sizes, and deletes it, and nothing else:
   - `.venv-cpu/` or `.venv-gpu/`: the private Python environment with PyTorch and its caches;
   - `models/wav2vec2-large-960h/`: the speech model downloaded at the start (1.2 GB);
   - temporary `__pycache__/` folders.

   Your data in `output/`, the scripts, `inputs/` and the YOLO weights are kept, and the machine is back to its previous state.

   The environment is **kept** only if the collection was interrupted (Ctrl+C, power cut), so that rerunning the same command resumes without reinstalling; `python3 run.py --cleanup` removes it if you don't want to resume. It is also kept after `--smoke`, because the full collection follows.
3. **Send back the zip** it creates: **`output/<machine_id>_<mode>.zip`**. It contains the folder `output/<machine_id>_<mode>/` with raw data, which will be combined with the data of the other machines and converted into a training format later (see `DOCUMENTATION.md`):

| File | Content |
|---|---|
| `files.csv.gz` | One row per image / audio clip: `Image No`, `Image Pixel`, `Processing Time`, `Execution Time (seconds)`, `CPU Usage Per Core`, `GPU Usage (%)`, … (the columns of the earlier datasets) plus raw columns |
| `requests.csv` | One row per request: execution, processing and waiting time of the whole folder, model load time, threads, resource use, hardware |
| `runs.csv` | One row per run: the columns of the earlier server CSVs (`Total Execution Time (sec)` = the slowest request's time, `Individual Execution Times`, `Peak RAM Usage (MB)`, …) plus status and checks |
| `monitor.csv.gz` | Raw CPU / RAM / GPU samples, about every 0.1 s |
| `machine_info.json` | Hardware, software and power settings |
| `collect.log`, `logs/` | Logs of the collection and of the whole session (setup included) |

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `could not create a virtual environment` | `sudo apt install python3-venv` (or the `python3.X-venv` it names) |
| `no usable Python 3.9 - 3.13 found` | Install Python 3.12 (python.org, or `sudo apt install python3.12 python3.12-venv`) |
| `not enough free disk space` | Free the amount it names, then run again |
| `GPU mode needs an NVIDIA GPU` / `no PyTorch CUDA build works on this GPU` | Check that `nvidia-smi` works; update the NVIDIA driver, or use `--mode cpu` |
| `running on battery` | Plug the machine in; runs wait until it is plugged in |
| "The machine is NOT ready" | Close the programs it lists, then press `r` to recheck |
| "Another collection from this folder ... is still running" | Answer `y` (default) to stop it; two collections at once would spoil each other's data |
| Runs marked `failed_oom` | Not enough RAM / GPU memory for that combination (e.g. on a 6 GB GPU the 4 combinations with four or more speech requests). This is recorded as the machine's capacity limit and the script continues |
| Runs marked `failed_busy` / `failed_suspended` | Another program was busy, the machine was on battery, or it slept. These are retried automatically; if some remain, rerun the same command |
| Interrupted / power cut | Rerun the same command; it repairs any half-written file and resumes |
| Need details of an error | Add `--verbose`, and look at `output/<machine_id>_<mode>/logs/` |

Other options: `python3 run.py --help` (launcher), `python3 run.py --help-collector` (collector, after setup), `--dry-run` (show the 55 combinations and the thread split).

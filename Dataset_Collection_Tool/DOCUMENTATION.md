# Computation-Latency Raw Data Collection

### Design, collection method and output reference

This document explains what the `Dataset_Collection_Tool` folder does, how the data is collected and why, and what every output file and column contains.
To simply run it, see **`README.md`** (quick start).

---

## At a glance

| | |
|---|---|
| **Goal** | Measure the computation delay of three edge services on a machine while 1–5 requests run on it at the same time, so that the scheduler can be trained and evaluated on heterogeneous edge servers |
| **Services** | `s` speech-to-text (Wav2Vec2-large) · `d` object detection (YOLOv5s) · `p` instance segmentation (YOLOv5s-seg) |
| **One request** | The task's whole input folder: 566 images (`d`, `p`) or 79 audio clips (`s`) |
| **How requests run** | Like requests reaching an edge server: every request has **its own process and its own model**. All requests of a combination arrive together; each loads its model and processes its folder straight away (**cold run**, no warm-up) |
| **CPU threads** | CPU mode: the machine's logical threads are **split evenly between the requests** of a run; leftover threads go to the slower tasks first (p > s > d); idle threads wait without spinning. GPU mode: PyTorch defaults |
| **Times recorded** | **Execution time** (wall clock of the whole folder) = **processing time** + **waiting time** (time the request was ready but had no free CPU). None includes the model load, which is stored in its own column |
| **What is measured** | Every combination of 1–5 concurrent requests (55 combinations) × 5 iterations = **275 runs**, **1,050 requests**, ~424,000 per-file measurements |
| **Modes** | GPU mode (NVIDIA GPU) or CPU-only mode; identical output format |
| **Different machines** | The setup detects each machine's hardware and installs the PyTorch build that works on it; the collection method is identical everywhere |
| **What the user does** | `python3 run.py` (or `run.sh` / `run.bat`), answer two questions, wait. Setup → collection → check → cleanup → zip happen automatically |
| **What comes back** | `output/<machine_id>_<mode>.zip`: raw data, combined later with the other machines' data and converted into a training format |

---

## Contents

1. [Purpose and context](#1-purpose-and-context)
2. [Folder contents](#2-folder-contents)
3. [The three services and their inputs](#3-the-three-services-and-their-inputs)
4. [Design principles](#4-design-principles)
5. [How a collection works](#5-how-a-collection-works)
6. [Setting up on different machines](#6-setting-up-on-different-machines)
7. [What is measured and how](#7-what-is-measured-and-how)
8. [Output files](#8-output-files)
9. [Column reference: `files.csv.gz`](#9-column-reference-filescsvgz)
10. [Column reference: `requests.csv`](#10-column-reference-requestscsv)
11. [Column reference: `runs.csv`](#11-column-reference-runscsv)
12. [From raw data to a training dataset](#12-from-raw-data-to-a-training-dataset)
13. [Robustness: clean, idle, powered, awake, resumable](#13-robustness-clean-idle-powered-awake-resumable)
14. [Code structure and options](#14-code-structure-and-options)
15. [Known limitations](#15-known-limitations)
16. [Comparison with the earlier dataset scripts](#16-comparison-with-the-earlier-dataset-scripts)

---

## 1. Purpose and context

Redundant scheduling (as in SafeTail) sends a user's service request to one or more edge servers to reduce **tail latency**. The latency of a request on server *eᵢ* is:

```
latency  =  transmission delay  +  propagation delay  +  computation delay C(eᵢ, sⱼ)
```

- Earlier work assumed identical (homogeneous) servers.
- This project targets **heterogeneous** edge servers: different CPUs, GPUs, core counts and memory.

The simulator therefore needs, for each kind of server, a realistic model of **computation delay**, including how it grows when several requests share the server.

- **Network delays** keep using the WiFi ping traces collected in earlier work.
- **This folder recollects the computation delay.** It is shared with the owners of about 5 different machines, some collecting in GPU mode and some CPU-only. Each runs the same script, and the per-machine datasets are combined into one heterogeneous dataset.

| Property | Meaning |
|---|---|
| **Like an edge server** | Each request is served independently and cold (first call included) |
| **Concurrency-aware** | Latency is measured while 1–5 requests share the machine |
| **Tail-aware** | Every request and every file is recorded, so distributions (p50, p90, p95, p99) can be computed |
| **Uniform** | Identical method, inputs and models on every machine; only the setup adapts to the hardware |
| **Raw and complete** | Everything is kept at file, request, run and ~0.1 s level; training formats are derived later |

---

## 2. Folder contents

```text
Dataset_Collection_Tool/
├── README.md                 quick start for the person running the script
├── DOCUMENTATION.md          this document
├── run.py                    the launcher: detection → setup → collection → check → cleanup → zip
├── run.sh / run.bat          one-command launchers for Linux / macOS and Windows
├── requirements.txt          pinned Python packages (PyTorch is chosen per machine, section 6)
├── collector/                the data-collection code (section 14)
├── yolov5/                   YOLOv5 code, used as a library (models/, utils/, export.py, LICENSE)
├── models/
│   ├── yolov5s.pt            detection weights
│   └── yolov5s-seg.pt        segmentation weights
├── inputs/
│   ├── images/               566 COCO images     = the detection / segmentation request folder
│   └── audio/                79 LibriSpeech clips = the speech request folder
└── output/                   collected data: one folder (and one .zip) per machine and mode
```

**Created temporarily on each machine and removed automatically at the end:**
- `.venv-cpu/` (about 1.5 GB) or `.venv-gpu/` (about 5–7 GB, includes NVIDIA's CUDA libraries): the private Python environment with all library caches;
- `models/wav2vec2-large-960h/` (1.2 GB): the speech model, downloaded on the first run.

### How YOLOv5 is used

The earlier scripts started `detect.py` / `segment/predict.py` as separate programs. Here the same YOLOv5 code is used **as a library** inside each request's process:

- the model is loaded with YOLOv5's own loader (`DetectMultiBackend`);
- every image goes through the **same steps** as `detect.py` / `segment/predict.py`: letterbox to 640, forward pass, NMS (confidence 0.25, IoU 0.45, max 1000 detections), box scaling, and masks (`process_mask`);
- annotated images are **not drawn or saved**, because a server returns results, not pictures.

Using it as a library is what makes the measurement possible: per-file timing, the model load measured separately, and all requests of a run arriving at the same moment. The YOLOv5 code itself is **unchanged** (nothing is patched). `yolov5/` contains only the parts needed to load and run the two models (0.7 MB instead of the 146 MB repository copy with training scripts, git history and old outputs).

---

## 3. The three services and their inputs

| Letter | Service | Model | Request = this folder | Result returned to the user |
|---|---|---|---|---|
| `s` | speech-to-text | Wav2Vec2-large-960h (315 M parameters) | `inputs/audio/`: 79 LibriSpeech test-clean clips, 16 kHz, 2.0–10.1 s each | transcripts |
| `d` | object detection | YOLOv5s (7.2 M parameters) | `inputs/images/`: 566 COCO images | boxes, confidence, class |
| `p` | instance segmentation ("predict") | YOLOv5s-seg (7.6 M parameters) | `inputs/images/`: 566 COCO images | boxes + pixel masks |

- These are the same models used by the earlier dataset scripts and the earlier project code.
- All models run in full precision (FP32).
- **Which task is slowest.** Each task was timed alone on the development laptop's CPU, processing its whole folder with the same number of threads (2 repetitions each, agreeing within 1–2 %):

  | Threads | `p` predict | `s` speech | `d` detect |
  |---|---|---|---|
  | 12 | 48.9 s | 46.4 s | 36.3 s |
  | 4 | 47.5 s | 43.0 s | 35.8 s |
  | 2 | 73.1 s | 72.0 s | 53.2 s |

  So per request the order is **p > s > d** at every thread count; this order decides who gets leftover CPU threads (section 7.3). Speech has by far the largest model (315 M parameters, ~1.3 GB) and the most work per file (~0.6 s per clip vs ~0.06–0.09 s per image), but its folder has only 79 files against 566 images. (On the GPU the order is also p first: 4.9 s, d 4.0 s, s 3.2 s alone.)
- Every machine processes **identical inputs**; `machine_info.json` stores a fingerprint of the input set so this can be checked.

---

## 4. Design principles

| # | Principle | Why |
|---|---|---|
| **P1** | **A request is a whole input folder** (566 images or 79 clips). Every file inside it is also logged separately. | Same request unit as the earlier dataset scripts; per-file detail is kept |
| **P2** | **Every request runs independently in its own process with its own model copy**, including same-type requests: `ss` = two speech processes, each with its own model. | Each request is served on its own, like independent server workers |
| **P3** | **Requests in a combination truly run in parallel and arrive together.** All request processes are started and made ready first; then all requests arrive at the same moment, and each loads its model and processes its folder. | The requests compete for the machine as concurrent requests on a server do |
| **P4** | **Execution time = processing time + waiting time**, all over the processing of the whole folder. **Model loading is not included** in any of them; it is stored in its own column. | Loading and processing can be analysed separately or added together |
| **P5** | **Cold runs, no warm-up:** the first file of every request is measured like the others, including first-call costs. | This happens on a server too. It can be identified later (`Image No` = 1) if needed |
| **P6** | **CPU mode: the machine's threads are shared out evenly between the requests of a run**; leftover threads go to the slower tasks first (p > s > d), and idle threads wait without spinning. GPU mode: PyTorch defaults. | Every logical thread is used, and the requests do not oversubscribe or waste the CPU (section 7.3) |
| **P7** | **Combinations are multisets:** `psp` = `pps` = `spp`, so 3 + 6 + 10 + 15 + 21 = **55 combinations**. | Requests running in parallel have no order |
| **P8** | **Every request is a target.** In `ssp`, the `s` rows give speech latency with `s` + `p` alongside; the `p` row gives predict latency with two `s` alongside. | No separate run per target task is needed |
| **P9** | **5 iterations, shuffled and spread over time.** | Slow drifts (heat, background activity) do not fall on one combination |
| **P10** | **System load is stored as peaks and averages** per file, request and run; the raw ~0.1 s trace is kept too. | The peak is what affects a request; averages match the earlier server CSVs |
| **P11** | **Measurements only on an idle machine on mains power.** Disturbed runs are discarded and repeated. | The final dataset contains only clean runs |
| **P12** | **Same method on every machine; setup adapts to the hardware.** | The newest PyTorch build that works on each machine (GPU-tested) is installed; the build is recorded |
| **P13** | **The machine is left as it was found, and kept awake.** | Nothing is installed system-wide; sleep cannot corrupt timings |
| **P14** | **Raw data first.** | The training format can change without re-collecting on five machines |
| **P15** | **Failures are data:** a combination that does not fit in memory is recorded as `failed_oom`. | It is the server's capacity limit |
| **P16** | **Safe to interrupt:** completed runs are kept; rerunning resumes, and anything half-written is repaired first. | Long collections survive interruptions and power cuts |

---

## 5. How a collection works

### 5.1 The whole flow: one command

```
python3 run.py          (or ./run.sh on Linux / macOS, run.bat on Windows)
   │
   ├─ 1. Detect the machine ... CPU, NVIDIA GPU + driver, other GPUs, Python, battery, free disk
   ├─ 2. Questions ............ cpu or gpu?  machine name?   (defaults taken after 10 min without an answer)
   ├─ 3. Setup ................ private environment, PyTorch build for this machine, packages, speech model
   ├─ 4. Before collecting .... stop leftovers of an earlier collection, repair half-written files,
   │                            check that the machine is idle and on mains power, block sleep
   ├─ 5. Collection ........... 275 runs, then up to 2 retry passes over runs that failed
   ├─ 6. Check ................ every run ok? one row per request, one row per file?
   ├─ 7. Cleanup .............. removes the environment and downloaded model (keeps output/)
   └─ 8. Zip .................. output/<machine_id>_<mode>.zip  → send this back
```

Everything shown on the screen (including installation output) is also written to `output/<machine_id>_<mode>/logs/launcher_<date>.log`.

### 5.2 The run plan

- **Combinations:** all 55 multisets of size 1–5 over {s, d, p}.
- **Iterations:** 5, in iteration-major order with the combinations shuffled each time (seed 2025 + iteration).
- **Total:** 275 runs and 1,050 requests (each run has k requests, k = 1…5).

### 5.3 One run, e.g. `ssp`, iteration 2, CPU mode on a 12-thread machine

```
 time ─────────────────────────────────────────────────────────────────────────────────────►
  pause 3 s │ wait until ready │ start processes │ ARRIVAL (t₀): all requests arrive together
                                                 │
 request s#0 (own process, 4 threads) ─ start up ┼─ load s model ─── 79 clips ──────► done
 request s#1 (own process, 4 threads) ─ start up ┼─ load s model ─── 79 clips ──────► done
 request p#0 (own process, 4 threads) ─ start up ┼─ load p model ── 566 images ──────────► done
                                                 │
 monitor (every ~0.1 s) ─────────────────────────┴──────────────────────────────────────────►
                                                    └ model load ┘└── execution time ──┘
```

1. **Pause** 3 s.
2. **Wait until ready** (up to 120 s): idle and on mains power. Otherwise the run is skipped as `failed_busy` and retried later (section 13).
3. **One process per request** starts (Python, PyTorch, the model code and, in GPU mode, the CUDA context) with its thread count, and reads the input features of its folder. Not timed: this is the server process being there.
4. When all are ready: **all requests arrive at the same moment (t₀).**
5. Each request **loads its own model** (timed: `model_load_time_s`).
6. Each request then **processes every file of its folder** one after another, **cold**. Every file is timed.
7. Other programs are watched throughout; a disturbed run is discarded and retried.
8. Peaks and averages are computed and the rows are written. `runs.csv` is written last, which marks the run as complete.

### 5.4 Duration

Measured on the development laptop (Intel i5-13420H, 8 cores / 12 threads, 16 GB RAM, RTX 4050 6 GB):

| Mode | Full collection (275 runs) |
|---|---|
| GPU | about **1.7 hours** |
| CPU only | about **10 hours** |

These are estimates from full-folder test runs of 6 combinations (`s`, `d`, `p`, `sdp`, `ssdpp`, `ddppp`) scaled to all 275 runs. A machine with fewer or slower cores takes proportionally longer in CPU mode (roughly up to twice as long on an older 4-core / 8-thread desktop), so both modes finish within a day. The collection can be stopped and resumed at any time.

---

## 6. Setting up on different machines

The **collection method is the same on every machine**; the **setup adapts** to the hardware. Before asking anything, `run.py` prints what it found:

```
  machine : linux-ubuntu  (Linux 6.14.0-37-generic, x86_64)
  CPU     : 13th Gen Intel(R) Core(TM) i5-13420H  (12 logical cores)
  GPU     : NVIDIA GeForce RTX 4050 Laptop GPU  6 GB, compute capability 8.9, driver 595.91.07 (CUDA <= 13.2)
  GPU     : Intel Corporation Raptor Lake-P [UHD Graphics]  (not NVIDIA - only CPU mode is possible with it)
  Python  : launcher 3.12.3; environment will use 3.12 (/usr/bin/python3)
  disk    : 31 GB free here, about 10 GB needed
```

| Step | What happens |
|---|---|
| **Machine detection** | CPU model and cores; NVIDIA GPUs with memory, compute capability, driver and the newest CUDA version the driver supports; other GPUs (AMD, Intel, Apple), which can only be used in CPU mode; OS; battery; free disk space (stops with a clear message if there is not enough) |
| **Questions** | Mode (default `gpu` when an NVIDIA GPU is present, otherwise `cpu`; GPU mode without an NVIDIA GPU is refused) and machine name (default: host name). Without an answer within 10 minutes, or without a keyboard, the default is taken, so an unattended start never hangs |
| **Python** | The best Python on the machine is used: 3.12 preferred, then 3.11, 3.10, 3.13, 3.9 (versions PyTorch publishes builds for). On Windows the `py` launcher is used; the Microsoft Store stub is skipped. If `venv` is missing, `virtualenv` is tried, and the exact install command is printed |
| **PyTorch, GPU mode** | The CUDA builds the driver supports are tried **newest first** (cu130, cu129, cu128, cu126, cu124, cu121, cu118). Each is installed and **tested on the GPU** with a real matrix multiplication and a cuDNN convolution. The first that works is kept; otherwise it is removed and the next older build is tried |
| **PyTorch, CPU mode** | The CPU build (macOS: the standard build) |
| **Other packages** | Installed from `requirements.txt` (pinned), keeping the chosen PyTorch build |
| **Check and repair** | All imports are tested, and in GPU mode the GPU test is repeated. On every later start the environment is checked again: broken packages are reinstalled; if PyTorch itself stopped working, the environment is rebuilt |
| **Speech model** | Downloaded once into `models/`; after that, collection runs offline |

Why the PyTorch build is chosen per machine and not pinned:
- **New GPUs** (e.g. RTX 50-series, compute capability 12.0) only work with recent CUDA builds.
- **Older GPUs** may be dropped by the newest builds.
- **Drivers limit** which CUDA builds can run.

The build used is recorded in `machine_info.json` and on every request and run row (`torch_version`, `torch_cuda`). On the development laptop the result was **PyTorch 2.14.1, CUDA 13.0**.

---

## 7. What is measured and how

### 7.1 Times

All durations are measured inside the request's process. In GPU mode, the GPU is synchronised before each timestamp, so a time includes the GPU work.

| Time | Request level | File level | Definition |
|---|---|---|---|
| **Execution time** | `execution_time_s` | `file_execution_ms` | Wall-clock time (`time.perf_counter`) from the start of the first file to the last result of the folder (file level: of that file). The time the job takes |
| **Waiting time** | `waiting_time_s` | `file_waiting_ms` | Time the request was **ready to run but had no free CPU core** (taken by other requests or programs) during that interval: the operating system's **run delay** of the request's main thread |
| **Processing time** | `processing_time_s` | `file_processing_ms` | **Execution − waiting**: the time the request was being processed (computing on the CPU, or on the GPU in GPU mode) |
| **Model load time** | `model_load_time_s` | — | Loading the request's model right after arrival. **Not part of any of the three times above** |

So **execution time = processing time + waiting time**, exactly, for every request and every file.

How the waiting time is measured:
- The Linux scheduler counts, for every thread, the time it was runnable but waiting in the queue for a CPU ("run delay", `/proc/thread-self/schedstat`). The collector reads it for the request's **main thread** before and after every file. The main thread takes part in every computation of the request (it runs its share of every multi-threaded operation and drives the GPU), so while it waits for a CPU the request is stalled.
- **On an idle machine** the waiting time of a single request is close to zero (about 3 % of the execution time on the development laptop); it grows when more threads compete for the CPU than there are hardware threads.
- **On Windows and macOS** the operating system does not report the run delay to Python. There, waiting time = execution time − CPU time of the main thread, which also counts moments when the main thread pauses for its own helper threads, so it is larger. `machine_info.json` → `waiting_time_method` records which method a machine used.

Related raw values (request level):
- `main_thread_cpu_time_s`: CPU time of the request's main thread;
- `process_cpu_time_s`: CPU time of **all** threads of the request (exceeds the execution time when several threads work at once; ÷ execution time = average number of busy threads);
- `threads_run_delay_s`: run delay summed over all threads of the request (Linux).

**In GPU mode**, while the GPU computes, the main thread waits for the result by polling it (CUDA's default), so it is running, not queued: the processing time includes the GPU's computation (and any wait for a GPU shared with other requests); the waiting time covers waits for a CPU.

Per file, the four stages are also recorded (they add up to `file_execution_ms`):

| Column | Stage |
|---|---|
| `read_ms` | Reading and decoding the file |
| `preprocess_ms` | Letterbox + tensor conversion (images) / feature normalisation (audio), copy to the device |
| `inference_ms` | The model's forward pass |
| `postprocess_ms` | NMS, box scaling, masks (segmentation) / CTC decoding (speech), copy of results back |

### 7.2 System monitoring: peaks and averages

A monitor in the main process samples the machine **about every 0.1 s** (`--sample-interval 0.1`; taking a sample takes time too, so on a fully loaded machine the actual spacing is 0.11–0.2 s; `t` in `monitor.csv.gz` has the exact times):
- total and per-core CPU %, CPU time split (user / system / disk wait);
- load average, RAM (MB and %), swap, CPU clock and temperature;
- disk read / write speed;
- GPU utilisation, memory-controller use, memory, clock, temperature and power;
- **once per second**, the CPU used by all other programs.

A sample at time *t* reports usage over the interval since the previous sample. The samples covering a time window are those inside it **plus the first one after it**, so even windows shorter than the sampling interval are covered.

For every **file** (its processing), every **request** (its folder processing) and every **run** (from the first request starting to process until the last one finishing), the collector stores:
- the **peak** of each quantity (`<quantity>_peak`);
- the **average** of CPU use, CPU clock, GPU use and GPU clock (`<quantity>_avg`);
- the peak and the average use of every logical core (`cpu_usage_per_core_peak`, `cpu_usage_per_core_avg`, JSON-style lists);
- `cores_used`: the number of cores whose **average** use was above 5 % (as in the earlier server CSVs).

### 7.3 Threads

**CPU mode.** The machine's logical threads (all hardware threads this process may use; e.g. 12 on an 8-core / 12-thread laptop) are the budget of every run. They are split between the requests of the run:

1. every request gets `budget ÷ k` threads (k = number of requests, at least 1);
2. the leftover threads go one each to the slowest requests first: **p > s > d** (section 3), then by request number.

| Combination | 8 threads | 12 threads |
|---|---|---|
| `s` | s 8 | s 12 |
| `spd` (= `sdp`) | p 3, s 3, d 2 | p 4, s 4, d 4 |
| `sssss` | 2, 2, 2, 1, 1 | 3, 3, 2, 2, 2 |
| `ssppd` | p 2, p 2, s 2, s 1, d 1 | p 3, p 3, s 2, s 2, d 2 |

The thread count is set in each request's process before PyTorch starts (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads`). Together the requests of a run use exactly the machine's threads: all of the CPU, without running more threads than there are hardware threads.

**Idle threads wait without spinning** (`OMP_WAIT_POLICY=PASSIVE`). By default, PyTorch's OpenMP threads keep a CPU busy ("spin") while they wait for the next piece of work. A spinning thread takes CPU time from the working threads (its hyper-thread sibling on the same core) and from the other requests of the run. On the development laptop (4 performance cores with hyper-threading + 4 efficiency cores), one speech request with 12 spinning threads was about **2× slower** than with passive waiting, and used 2.6× more CPU time. With passive waiting, waiting threads sleep and wake up when work arrives.

**GPU mode.** PyTorch's defaults are used (the computation runs on the GPU; the CPU threads only prepare inputs and collect results).

The values used are recorded: `threads_allocated` and `torch_threads` per request, `Number of Threads` per file, `Threads Per Script` per run, and the policy in `machine_info.json` (`thread_settings`).

---

## 8. Output files

Every collection writes one folder, `output/<machine_id>_<mode>/`, and at the end packs it into `output/<machine_id>_<mode>.zip`:

```text
output/<machine_id>_<mode>/
├── files.csv.gz        one row per image / audio clip   ← per-file data, with the earlier datasets' columns
├── requests.csv        one row per request (one folder) ← execution / processing / waiting time per request
├── runs.csv            one row per run                  ← the earlier server CSV columns + quality control
├── monitor.csv.gz      one row per ~0.1 s sample        ← raw system trace
├── machine_info.json   one record                       ← hardware, software, power settings
├── collect.log         text log of the collector        ← every run's result, failures, retries, final check
└── logs/
    └── launcher_<date>.log   everything shown on the screen, including installation output
```

### 8.1 The files in brief

| File | One row per | Rows (full run) | Used for |
|---|---|---|---|
| `files.csv.gz` | image / audio clip processed | ~424,000 | Per-file data in the **column layout of the earlier datasets** (`Image No`, `Processing Time`, `Execution Time (seconds)`, `CPU Usage Per Core`, …) plus raw columns. Most samples for tail analysis |
| `requests.csv` | request (one task's whole folder in one run) | 1,050 | **Execution, processing and waiting time** of each request under each combination, model load time, threads, process counters, peaks / averages, hardware |
| `runs.csv` | run (one combination × one iteration) | 275 (+ repeated runs) | **All columns of the earlier server CSVs** (`Total Processing Time (sec)`, `Individual Execution Times`, `Peak RAM Usage (MB)`, `Average CPU Usage Per Core`, …) plus status, idle and power checks |
| `monitor.csv.gz` | system sample (~0.1–0.2 s apart) | about 5–10 per second of collection | Raw evidence behind the peaks and averages; other statistics can be computed later |
| `machine_info.json` | machine | 1 | CPU / GPU model, cores, RAM, driver, OS, Python, PyTorch build, power settings, thread policy, input fingerprint |
| `collect.log` | log line | — | Human-readable record: machine, idle checks, every run's times, failures, retries, final check |
| `logs/launcher_<date>.log` | log line | — | Detection, setup and installation output, the collection, cleanup (one file per start) |

`files.csv.gz` and `monitor.csv.gz` are gzip-compressed CSV files (`pandas.read_csv` opens them directly).

### 8.2 How the files link together

| Link | Key |
|---|---|
| Everything in one folder | `machine_id` + `device_mode` |
| `runs.csv` ↔ `requests.csv` ↔ `files.csv.gz` ↔ `monitor.csv.gz` | `run_id` (e.g. `ssp_it1`) |
| A request ↔ its files | `run_id` + `task` + `slot` (files in the order of `Image No`) |
| Data ↔ quality | Only runs with `status == ok` have rows in `requests.csv` / `files.csv.gz` |

Only durations are stored for requests and files (no clock timestamps); the peaks and averages over their time windows are computed by the collector.

### 8.3 `monitor.csv.gz` columns

| Column | Meaning |
|---|---|
| `run_id`, `t`, `phase` | Run, Unix time of the sample, `load` (request processes starting up) or `run` (after arrival: model loads and processing) |
| `cpu_util`, `cpu_per_core` | Total CPU %, and % of every logical core (semicolon-separated) |
| `cpu_user_pct`, `cpu_system_pct`, `cpu_iowait_pct` | CPU time in programs / in the operating system / waiting for disk |
| `load_avg_1m` | 1-minute load average |
| `ram_used_mb`, `ram_used_pct`, `swap_used_mb` | Memory in use (MB and %), swap |
| `cpu_freq_mhz`, `cpu_temp_c` | CPU clock, hottest CPU temperature sensor |
| `disk_read_mb_s`, `disk_write_mb_s` | Disk throughput |
| `gpu_util`, `gpu_mem_ctrl_util`, `gpu_mem_used_mb`, `gpu_clock_mhz`, `gpu_temp_c`, `gpu_power_w` | GPU state (empty in CPU mode) |
| `foreign_cpu_pct`, `foreign_top`, `foreign_gpu` | CPU % used by other programs, the largest of them by name, other programs on the GPU |

### 8.4 `machine_info.json`

- `machine_id`, `device_mode`, `hostname`, `os`, `python`;
- `cpu_model`, `cpu_physical_cores`, `cpu_logical_cores`, `cpu_max_freq_mhz`, `ram_total_gb`;
- `gpu_model`, `gpu_mem_total_gb`, `gpu_driver`, `gpu_max_sm_clock_mhz`, `gpu_count` (in CPU mode an unused GPU is listed as `gpu_present_but_unused`);
- `torch_version`, `torch_cuda`, `cudnn`, `cuda_available`, `torch_default_threads`, `transformers_version`;
- power: `on_battery_at_start`, `cpu_governor`, `cpu_energy_preference`, `power_profile` (recorded, not changed);
- `inputs_fingerprint` (identical on every machine when the inputs are identical);
- `waiting_time_method`: how the waiting time was measured (section 7.1);
- `format_version` (`2026-10-06` = the current layout), `thread_settings` (CPU mode: the thread budget, the split rule and `OMP_WAIT_POLICY=PASSIVE`; GPU mode: PyTorch defaults), `files_per_request`, `warmup` (always "none (cold runs)").

### 8.5 Example: run `ssdpp_it1`, laptop, CPU mode (smoke test, 20 files per request, 12 threads)

`requests.csv`:

| task | slot | threads | model_load_time_s | execution_time_s | processing_time_s | waiting_time_s | running_at_end |
|---|---|---|---|---|---|---|---|
| s | 0 | 2 | 0.566 | 21.479 | 21.436 | 0.043 | 1 |
| s | 1 | 2 | 0.684 | 20.843 | 20.817 | 0.026 | 2 |
| d | 0 | 2 | 0.769 | 4.594 | 4.526 | 0.068 | 5 |
| p | 0 | 3 | 0.742 | 4.926 | 4.848 | 0.079 | 4 |
| p | 1 | 3 | 0.774 | 4.989 | 4.893 | 0.096 | 3 |

- The 12 threads were split 2 + 2 + 2 + 3 + 3: the two leftover threads went to the predict requests (the slowest task, section 3).
- Each request loaded its own model after arrival; the load is in its own column.
- With the even split the requests hardly wait for a CPU (waiting time < 2 %): each has its own share of the CPU, and the effect of running together shows in longer execution times than alone.

---

## 9. Column reference: `files.csv.gz`

One row per image or audio clip processed (60 columns). The first block has the **column names of the earlier datasets**.

### 9.1 Columns of the earlier datasets

| Column | Images (`d`, `p`) | Audio (`s`) |
|---|---|---|
| `Image No` | Position of the file in its request's folder (1…566) | Position of the clip (1…79) |
| `Image Pixel` | Resolution `width x height`, e.g. `640x426` | empty |
| `Processing Time` | **Processing time of this file (ms)** = `file_processing_ms` | same |
| `Iteration` | Repetition number (1–5) | same |
| `Combination` | The requests that ran together, e.g. `ssp` | same |
| `Current Script` | `detect.py` or `predict.py` | `speech.py` |
| `Execution Time (seconds)` | **Execution time of the whole request** this file belongs to (= `execution_time_s` in `requests.csv`); the same on all files of the request | same |
| `RAM Memory Usage (MB)` | Memory used by this request's process after this file | same |
| `CPU Usage Per Core` | Peak use of every logical core while this file was processed (list) | same |
| `GPU Usage (%)` | Peak GPU utilisation while this file was processed (empty in CPU mode) | same |
| `GPU Memory Usage (MB)` | GPU memory held by this request's process (empty in CPU mode) | same |
| `Total RAM (GB)` | Installed RAM | same |
| `Total CPU Cores` | Logical cores (hardware threads) of the machine | same |
| `Total GPU Memory (GB)` | GPU memory (empty in CPU mode) | same |
| `CPU Model`, `GPU Model` | Processor and GPU names (GPU empty in CPU mode) | same |
| `CPU Clock Speed (MHz)` | Average CPU clock while this file was processed | same |
| `GPU Clock Speed (MHz)` | Average GPU clock while this file was processed (empty in CPU mode) | same |
| `Number of Cores Used` | Cores whose average use was above 5 % while this file was processed | same |
| `File` | File name | same |
| `Duration` | empty | Clip length in seconds |
| `Number of Threads` | PyTorch threads of the request | same |
| `Physical CPU Cores` | Physical cores of the machine | same |
| `Total RAM Used (MB)` | Peak RAM in use on the whole machine while this file was processed | same |
| `RAM Utilisation (%)` | The same as a percentage of installed RAM | same |

### 9.2 Raw columns

| Column | Meaning |
|---|---|
| `machine_id`, `device_mode` | Machine name; `gpu` or `cpu` |
| `run_id`, `concurrency` | Run (`<combination>_it<iteration>`), number of requests in it (k) |
| `task`, `task_name`, `slot` | Task (`s`/`d`/`p`, `speech`/`detect`/`predict`) and number among same-type requests (0, 1, …) |
| `input_bytes`, `img_width`, `img_height`, `img_pixels`, `audio_samples` | Size of the file; resolution (images) or number of samples (audio) |
| `file_processing_ms`, `file_execution_ms`, `file_waiting_ms` | Processing, execution and waiting time of this file (section 7.1); processing + waiting = execution |
| `read_ms`, `preprocess_ms`, `inference_ms`, `postprocess_ms` | Stages of this file (section 7.1) |
| `n_objects`, `output_bytes` | Detections / masks / words found, and the size of the result |
| `requests_running` | Requests of the run (this one included) that had not yet finished when this file finished. Equal to `concurrency` until the first request finishes; lower afterwards |
| `cpu_util_peak`, `cpu_util_avg` | Total CPU use (% of all cores) while this file was processed |
| `cpu_freq_mhz_peak`, `cpu_temp_c_peak`, `cpu_iowait_pct_peak`, `load_avg_1m_peak`, `swap_used_mb_peak` | Peak CPU clock, CPU temperature, disk wait, load average, swap |
| `gpu_util_avg`, `gpu_mem_used_mb_peak`, `gpu_clock_mhz_peak`, `gpu_temp_c_peak`, `gpu_power_w_peak` | GPU state (empty in CPU mode) |
| `n_monitor_samples` | Number of monitor samples behind the peaks / averages |

---

## 10. Column reference: `requests.csv`

One row per request (77 columns).

### 10.1 Which run and which request

| Column | Meaning |
|---|---|
| `machine_id` | Name given to the machine when the script was started |
| `device_mode` | `gpu` or `cpu` |
| `run_id` | `<combination>_it<iteration>`, e.g. `ssp_it1` |
| `combination` | The requests that ran together, letters in the order s, d, p |
| `concurrency` | k = number of requests in the run (1–5) |
| `n_s`, `n_d`, `n_p` | Number of speech / detect / predict requests in the combination |
| `iteration` | Repetition number (1–5) |
| `task`, `task_name`, `script` | This request's type: `s` speech (`speech.py`), `d` detect (`detect.py`), `p` predict (`predict.py`) |
| `slot` | Number of this request among the same-type requests of the run (0, 1, …) |
| `others_s`, `others_d`, `others_p` | Requests of each type running alongside this one |

### 10.2 Times

| Column | Meaning |
|---|---|
| **`execution_time_s`** | **Execution time**: wall-clock time to process the whole folder, first file start → last result (s) |
| **`waiting_time_s`** | **Waiting time**: time the request was ready to run but had no free CPU (s; section 7.1) |
| **`processing_time_s`** | **Processing time** = execution − waiting (s) |
| `model_load_time_s` | Time to load this request's model right after arrival (s). Not included in the three times above |
| `running_at_end` | Requests of this run not yet finished when this one finished, itself included (k = finished first, 1 = finished last) |

### 10.3 Per file of this request

| Column | Meaning |
|---|---|
| `n_files` | Files in the request: 79 (speech) or 566 (detect / predict) |
| `file_execution_ms_mean`, `file_execution_ms_p50`, `file_execution_ms_p95`, `file_execution_ms_p99`, `file_execution_ms_max` | Distribution of the per-file execution times **inside this request** (ms) |
| `file_inference_ms_mean` | Mean forward-pass time per file (ms) |

### 10.4 Input and output

| Column | Meaning |
|---|---|
| `input_bytes_total` | Total size of the input files in bytes (the `Size(I)` term) |
| `img_pixels_total` | Total pixels of all images (empty for speech) |
| `audio_duration_s_total` | Total audio length in seconds (empty for images) |
| `first_img_pixels`, `first_audio_samples` | Size of the first file (pixels / samples) |
| `n_objects_total` | Detections / masks found or words transcribed, over all files |
| `output_bytes_total` | Approximate size of all results (the `eSize(O)` term) |

### 10.5 The request's process

| Column | Meaning |
|---|---|
| `threads_allocated` | Threads given to this request (CPU mode, section 7.3; empty in GPU mode = PyTorch default) |
| `torch_threads` | PyTorch threads actually used |
| `main_thread_cpu_time_s` | CPU time of the request's main thread while processing the folder |
| `process_cpu_time_s` | CPU time of **all** threads of the process while processing the folder (can exceed the execution time; ÷ execution time = average number of busy threads) |
| `threads_run_delay_s` | Run delay (waiting for a CPU) summed over all threads of the process (Linux; empty elsewhere) |
| `process_ctx_switches_involuntary` | Times the OS took a CPU away from the process, a direct sign of CPU contention |
| `process_peak_rss_mb` | Peak memory of the process |
| `process_gpu_peak_mb` | Peak GPU memory reserved by the process (empty in CPU mode) |
| `model_load_disk_read_mb` | MB read from disk while loading the model (≈ 0 when the file was already in memory) |

### 10.6 System use while the request processed its folder

System-wide, from the monitor samples, from the request's first file to its last result. GPU columns are empty in CPU mode.

| Column | Meaning |
|---|---|
| `cpu_util_peak`, `cpu_util_avg` | Total CPU use (% of all cores): peak / average |
| `cpu_usage_per_core_peak`, `cpu_usage_per_core_avg` | Use of every logical core: peak / average (lists) |
| `cores_used` | Cores whose average use was above 5 % |
| `cpu_freq_mhz_peak`, `cpu_freq_mhz_avg` | CPU clock (MHz): peak / average |
| `cpu_temp_c_peak` | Hottest CPU temperature (°C) |
| `cpu_iowait_pct_peak` | CPU % waiting for the disk |
| `load_avg_1m_peak` | 1-minute load average |
| `ram_used_mb_peak`, `ram_used_pct_peak` | RAM in use on the machine (MB, %) |
| `swap_used_mb_peak` | Swap in use (MB) |
| `gpu_util_peak`, `gpu_util_avg` | GPU compute use (%): peak / average |
| `gpu_mem_used_mb_peak` | GPU memory in use, all processes (MB) |
| `gpu_clock_mhz_peak`, `gpu_clock_mhz_avg` | GPU clock (MHz): peak / average |
| `gpu_temp_c_peak`, `gpu_power_w_peak` | GPU temperature (°C), power (W) |
| `n_monitor_samples` | Number of monitor samples behind these values |

### 10.7 Machine and software description

Identical on every row of a machine.

| Column | Meaning |
|---|---|
| `cpu_model` | CPU name |
| `cpu_physical_cores`, `cpu_logical_cores` | Physical and logical cores |
| `cpu_max_freq_mhz` | Maximum CPU clock |
| `ram_total_gb` | Installed RAM |
| `gpu_model`, `gpu_mem_total_gb`, `gpu_driver` | GPU name, memory and NVIDIA driver (empty in CPU mode) |
| `os`, `python` | Operating system and Python version |
| `torch_version`, `torch_cuda` | PyTorch build and its CUDA version (empty in CPU mode) |

---

## 11. Column reference: `runs.csv`

One row per run (67 columns). A run that was repeated (busy, slept, error) appears more than once; its **last** row is its final status. A failed attempt writes only its line here: `requests.csv` and `files.csv.gz` contain rows only of the attempt that ended `ok`. The first block has **all columns of the earlier server CSVs** (`server1.csv` in GPU mode, `server3.csv` in CPU mode) except the source-folder paths, which are the same bundled folders everywhere.

### 11.1 Columns of the earlier server CSVs

Lists such as `Individual Execution Times` are comma-separated, one value per request, in the order of `Scripts Executed`. Empty for runs that are not `ok`.

| Column | Meaning |
|---|---|
| `Iteration`, `Combination` | Repetition number, combination (e.g. `ssp`) |
| `Total Execution Time (sec)` | Execution time of the run = execution time of its **slowest request**. The requests run in parallel, so the run is done when its slowest request is done. Model load not included |
| `Total Processing Time (sec)` | Processing time of that same (slowest) request |
| `Total Waiting Time (sec)` | Waiting time of that same request, so `Total Execution Time` = `Total Processing Time` + `Total Waiting Time` |
| `Scripts Executed` | The requests, e.g. `Speech, Speech, Predict` |
| `Individual Processing Times`, `Individual Execution Times` | Processing / execution time of every request (s) |
| `Files Processed Per Script` | Files of every request, e.g. `79, 79, 566` |
| `Num Files Processed` | Files of all requests together |
| `First Image Pixels`, `First Audio Samples` | Size of the first image / clip (empty when the run has no image / speech request) |
| `Peak RAM Usage (MB)` | Peak RAM in use on the machine |
| `Peak CPU Usage (%)` | Peak total CPU use |
| `Peak GPU Usage (%)`, `Peak GPU Memory (MB)` | Peak GPU use and GPU memory in use (empty in CPU mode) |
| `Average CPU Usage Per Core` | Average use of every logical core (list) |
| `Number of Cores Used` | Cores whose average use was above 5 % |
| `Average CPU Clock (MHz)`, `Average GPU Clock (MHz)` | Average clocks (GPU empty in CPU mode) |
| `Total RAM (GB)`, `Total CPU Cores`, `Total GPU Memory (GB)`, `CPU Model`, `GPU Model` | Machine description |
| `Device` | `GPU` or `CPU` |
| `Individual Waiting Times`, `Individual Model Load Times` | Waiting time and model load time of every request (s), same style |
| `Threads Per Script` | PyTorch threads of every request |

The earlier server CSVs summed the scripts' times, because their benchmark ran the scripts one after another. Here the requests run at the same time, so a sum would overstate the run's time (up to k times); the per-request times are in the `Individual …` lists and in `requests.csv`.

Peaks and averages cover the processing part of the run: from the first request starting to process its folder to the last one finishing.

### 11.2 Collection details

| Column | Meaning |
|---|---|
| `machine_id`, `device_mode`, `run_id`, `concurrency`, `n_s`, `n_d`, `n_p` | Which run |
| `status` | `ok` = measured and saved · `failed_busy` = machine not idle or on battery before, or disturbed during, the run (discarded, retried) · `failed_oom` = did not fit in RAM / GPU memory (capacity limit) · `failed_suspended` = the machine slept (discarded, retried) · `failed` = another error |
| `error` | The reason, when the status is not `ok` |
| `started_at` | Date and time the run started |
| `pre_run_cpu_util`, `pre_run_foreign_cpu` | Total CPU %, and CPU % of other programs, just before the run |
| `idle_wait_s` | Seconds waited for the machine to become ready |
| `busy_processes` | Other programs using ≥ 2 % of the CPU before the run |
| `foreign_cpu_avg`, `foreign_cpu_max` | Average / highest CPU % used by other programs during the run (≤ 5 % average in an `ok` run) |
| `foreign_top_processes` | Other programs most often active during the run |
| `foreign_gpu_processes` | Other programs that used the GPU during the run (empty in an `ok` run) |
| `baseline_ram_used_mb`, `baseline_gpu_mem_used_mb` | Memory in use before the run |
| `startup_s` | Seconds from starting the request processes until all were ready (before arrival; not part of any request time) |
| `run_time_s` | Arrival (t₀) → last request done (s): how long the whole run took, **model loads included** = the largest (model load + execution time) over the run's requests. It differs from `Total Execution Time (sec)` by the model load (and, when another request loaded faster but ran shorter, by that request). The start-up of the request processes before arrival (`startup_s`) is not included |
| `requests` | Request rows written (0 if not `ok`) |
| `suspended_s` | Seconds the machine was asleep during the run (0 normally) |
| `swap_in_mb` | Data read back from swap during the run; more than a few MB means RAM ran short |
| `cpu_util_avg`, `cpu_usage_per_core_peak`, `cpu_freq_mhz_peak`, `cpu_temp_c_peak`, `cpu_iowait_pct_peak`, `load_avg_1m_peak`, `ram_used_pct_peak`, `swap_used_mb_peak`, `gpu_util_avg`, `gpu_clock_mhz_peak`, `gpu_temp_c_peak`, `gpu_power_w_peak` | Further peaks and averages over the run (as in section 10.6) |
| `duration_s` | Total wall time of the run including pause, idle wait, start-up, model loads and saving |

---

## 12. From raw data to a training dataset

This folder **only collects raw data**. Turning it into training data for the scheduler is a separate, later step, done once the data of all machines is in.

```
 machine 1 ─┐
 machine 2 ─┤   run.py on each       <machine>_<mode>.zip           ① combine       ② check        ③ convert
 machine 3 ─┼─────────────────────►  raw data, identical format  ─────────────► ─────────────► ─────────────►  training data
 machine 4 ─┤   (setup per machine,                                all machines    clean runs     format designed
 machine 5 ─┘    same method)                                      in one table    only           with the 2.0 code
```

**① Combine.** Unzip all returned files into one `output/` directory and stack each file type; `machine_id`, `device_mode` and the hardware columns keep the machines apart:

```python
import glob, json, pandas as pd

files    = pd.concat(pd.read_csv(f) for f in glob.glob("output/*/files.csv.gz"))
requests = pd.concat(pd.read_csv(f) for f in glob.glob("output/*/requests.csv"))
runs     = pd.concat(pd.read_csv(f) for f in glob.glob("output/*/runs.csv"))
machines = pd.DataFrame(json.load(open(f)) for f in glob.glob("output/*/machine_info.json"))

assert machines["inputs_fingerprint"].nunique() == 1    # every machine processed identical inputs
assert machines["format_version"].nunique() == 1        # every machine used the same version of this tool
```

**② Check.**
- Rows exist only for runs with `status == ok`; disturbed, interrupted and half-written runs were already removed and repeated (the final check in `collect.log` confirms one row per request and per file).
- `failed_oom` runs mark what does not fit on that machine.
- For per-file tail analysis under full concurrency, keep files with `requests_running == concurrency` (no request of the run had finished yet). To study warm behaviour only, drop `Image No == 1`.

**③ Convert.** The format is designed together with the scheduler code. For example:

- **Per-request times** (`requests.csv`): `execution_time_s`, `processing_time_s`, `waiting_time_s`, with `model_load_time_s` available separately for cold-start modelling.
- **The earlier server-CSV layout** (`dataset/server<N>.csv`): `runs.csv` already has its columns. Its ordered combination strings (e.g. `psp`) are filled from the run of their multiset (`spp`).
- **Per-file tail latency** per machine, combination and task:
  ```python
  full = files[files.requests_running == files.concurrency]
  tail = (full.groupby(["machine_id", "device_mode", "Combination", "task"])["file_execution_ms"]
              .quantile([0.5, 0.9, 0.95, 0.99]).unstack())
  ```
- **A regressor of computation delay.**
  - Target: `execution_time_s` (request) or `file_execution_ms` (file).
  - Features: task, co-runners (`others_*`, `concurrency`), threads, input size (`img_pixels`, `audio_duration_s`, `input_bytes`), hardware columns, resource use.

---

## 13. Robustness: clean, idle, powered, awake, resumable

### 13.1 Nothing is left behind

- **Everything is installed inside this folder** (`.venv-cpu/` or `.venv-gpu/`); nothing system-wide, no administrator rights.
- **No caches outside the folder:** pip keeps no download cache; Hugging Face, PyTorch, matplotlib, Ultralytics, CUDA and other caches / settings are redirected into the environment folder; no `__pycache__` is written.
- **Automatic cleanup:** when the collection has finished (including retries), `run.py` lists what it installed or downloaded and deletes it without asking:

| Deleted | Size | What it is |
|---|---|---|
| `.venv-cpu/` and / or `.venv-gpu/` | ~1.5 GB / ~5–7 GB | Private Python environment (PyTorch, transformers, …; GPU version includes CUDA libraries) with all caches |
| `models/wav2vec2-large-960h/` | 1.2 GB | Speech model downloaded at the start |
| `__pycache__/` folders | small | Temporary Python files |

It **keeps** `output/` (data and zip), the scripts, `inputs/` and the YOLO weights.

The environment is kept only:
- if the collection was **interrupted** (Ctrl+C, power cut): rerunning resumes without reinstalling, and `python3 run.py --cleanup` removes it instead;
- after **`--smoke`** / **`--dry-run`**, because the full collection follows;
- with **`--keep-env`**.

### 13.2 Nothing else runs during measurement

"Other programs" are all processes except the collector, its request processes and its launcher. A machine is **ready** when:
- total CPU ≤ 10 %;
- other programs use ≤ 5 % of the CPU;
- in GPU mode, no other program is on the GPU and GPU use ≤ 10 %;
- it runs on **mains power** (a laptop on battery slows down to save power).

| When | What happens |
|---|---|
| **At start** | Leftovers of an earlier collection from this folder (a collection still running, or request processes of one that was killed) are found and stopped. Two collections at once would disturb each other |
| **Before collection** | 5 s check. If not ready, the reasons and the programs are listed by name. The user can recheck, continue or quit; without an answer it rechecks every 5 minutes (without a keyboard every minute) and continues after 6 checks; runs are then only measured once the machine is ready |
| **Before every run** | Wait up to 120 s until ready; if not, the run is **not measured** (`failed_busy`, retried later) |
| **During every run** | Once per second, other programs' CPU and GPU use are measured and stored (`foreign_*`). If they averaged > 5 % of the CPU, another program used the GPU, or the machine switched to battery, the run is **discarded** (`failed_busy`, retried later) |
| **After the main pass** | Every run that is not `ok` (except `failed_oom`) is run again, up to 2 extra passes |

### 13.3 The machine stays awake

- **Sleep is blocked while collecting.**
  - Linux: `systemd-inhibit`, which also blocks suspend on closing the lid, without administrator rights (where the lid cannot be blocked, sleep and idle-suspend still are).
  - macOS: `caffeinate`.
  - Windows: the keep-awake flag.
- **Sleep is detected anyway.** A run during which the machine was suspended for more than 2 s is discarded (`failed_suspended`) and redone.
- **Shutting down** stops the collection; rerunning the same command resumes from the last completed run.

### 13.4 Interruptions and half-written files

- `runs.csv` is written **last** for every run, so a run counts as done only when its line there is complete.
- Before resuming, the collector **repairs** the output: a line cut off by a power cut is removed, and rows of a run that never finished are dropped from `requests.csv` / `files.csv.gz` (the run is then measured again). Afterwards every row belongs to exactly one completed run.
- A dataset folder never mixes settings: if the machine, mode, inputs, thread policy or version of this tool differ from those the folder was started with, the collector stops with a message. Data from an earlier version of this tool is kept under `<folder>_old_<date>` and a fresh folder is started.

### 13.5 Final check and zip

When the collection ends, the collector checks that every planned run is `ok` (or `failed_oom`), that every `ok` run has one row per request in `requests.csv`, and that every request has one row per file in `files.csv.gz`. The result is printed and written to `collect.log`. Then `run.py` cleans up and packs the output folder into **`output/<machine_id>_<mode>.zip`**, the file to send back.

---

## 14. Code structure and options

| File | Responsibility |
|---|---|
| `run.py` | Launcher (standard library only): machine detection, questions with time-out, disk check, Python selection, per-machine PyTorch selection with GPU test, packages, check / repair of the environment, speech model, keep-awake, collection, cleanup, zip, launcher log |
| `run.sh`, `run.bat` | One-command launchers that find a suitable Python and start `run.py` |
| `collector/collect.py` | Run plan, thread split, leftover / idle / power / sleep checks, runs (one process per request, common arrival, results), peaks and averages, output files, repair, resume, retries, final check |
| `collector/worker.py` | One request: start-up with its thread count, wait for arrival, model load, process the folder cold, measure execution / processing / waiting times, report results or errors |
| `collector/tasks.py` | The three services with timed stages; input-feature reader |
| `collector/monitor.py` | ~0.1 s system sampler (psutil + NVIDIA NVML), other-program tracking, idle check, battery |
| `collector/sysinfo.py` | Hardware / software / power description and input fingerprint |
| `collector/config.py` | Task letters, names, input folders, output format version |
| `collector/prompt.py` | Questions with a time-out and a default |
| `collector/download_models.py` | Downloads the speech model into `models/` |

**Launcher options** (`python3 run.py --help`):

| Option | Meaning |
|---|---|
| `--mode cpu\|gpu`, `--machine-id NAME` | Answer the two questions on the command line |
| `--smoke` | 2–5 minute check: 5 short runs (20 files per request) into a separate folder |
| `--dry-run` | Print the 55 combinations and the thread split, then exit |
| `--cleanup` | Remove the environment and downloaded model now (`--yes` skips the confirmation) |
| `--keep-env` | Do not clean up automatically at the end |
| `--setup-only`, `--reinstall` | Only install / rebuild the environment |
| `--help-collector` | Show the collector options below |

**Collector options** (advanced; all machines should use the defaults):

| Option | Default | Meaning |
|---|---|---|
| `--iterations` | 5 | Repetitions per combination |
| `--files-per-request` | 0 | 0 = whole folder; N = a seeded random subset of N files |
| `--max-concurrency` | 5 | Largest combination size |
| `--combos` | all | Explicit list, e.g. `s,dp,spp` |
| `--cpu-threads` | 0 | CPU mode: thread budget per run (0 = all logical threads) |
| `--idle-threshold`, `--foreign-threshold`, `--idle-wait` | 10 %, 5 %, 120 s | Idle rules (section 13.2) |
| `--allow-battery` | off | Measure even on battery |
| `--retry-passes` | 2 | Extra passes over failed runs |
| `--cooldown`, `--sample-interval` | 3 s, 0.1 s | Pause between runs; monitor period |
| `--run-timeout`, `--ready-timeout` | 7200 s, 900 s | Limits for one run and for the request processes to start |
| `--verbose` | — | Show error tracebacks |

---

## 15. Known limitations

- **One model copy per request** uses more memory: five speech requests load five 1.3 GB models. In CPU mode on the development laptop (16 GB RAM) `sssss` fits, with about 13 GB of RAM in use. In GPU mode each speech request holds about 1.3–1.45 GB of GPU memory: on the laptop's 6 GB GPU, `sss` + up to two image requests fit (peak 5.4–6.0 GB), while the 4 combinations with four or more speech requests (`ssss`, `ssssd`, `ssssp`, `sssss`, i.e. 20 of 275 runs) are recorded as `failed_oom`. Combinations that do not fit are recorded as `failed_oom` (the machine's capacity limit); paging to swap is visible in `swap_in_mb`.
- **Processing time in GPU mode** includes the time the main thread waits for the GPU's result (section 7.1), so the waiting time there only reflects waits for a CPU.
- **Waiting time on Windows / macOS** is measured with CPU time instead of the run delay (section 7.1), and per-file values there are coarser (Windows counts CPU time in steps of about 16 ms). Request execution times are not affected.
- **Waiting time follows the main thread.** A helper thread waiting for a CPU slows the request down without being counted as waiting; `threads_run_delay_s` shows it.
- **PyTorch builds differ between machines** (the newest build that works on each). The build is recorded on every row.
- **Same input in every request of a task.** Every request processes the same folder; the effect of image size or clip length is in `files.csv.gz`.
- **5 samples per (combination, task)** at request level. For tails, use the per-file data (`files.csv.gz`, hundreds of files per request).
- **Model files are usually read from memory** after the first run (OS file cache), so `model_load_time_s` mostly measures building the model; `model_load_disk_read_mb` shows which case occurred.
- **Monitor resolution is about 0.1–0.2 s** (configured 0.1 s; other programs: 1 s). Very short files are covered by 1–3 samples, so their peaks describe the surrounding 0.1–0.2 s. NVIDIA's utilisation counter is itself averaged by the driver.
- **CPU clock is the average over all logical cores, idle ones included** (as the operating system reports it), so in GPU mode, where most cores are idle, it reads low (e.g. 1.0–2.3 GHz on a Ryzen 9 7950X). On Windows the reported clock is the fixed base clock, and CPU temperature, disk wait and swap-in are not available.
- **Laptops heat up** during long CPU collections and lower their clock; this is part of the machine's behaviour and visible in `cpu_temp_c_peak` / `cpu_freq_mhz_avg`.
- **Speech inputs come from one LibriSpeech chapter (one reader)**, as available in the shared material.
- **Operating-system background services** cannot be stopped. Their CPU use is measured; runs where it exceeds 5 % are discarded and repeated.

---

## 16. Comparison with the earlier dataset scripts

| Aspect | Earlier (`Script_for_DATASET_GENERATION`, `Regressor_*` benchmarks) | This collector |
|---|---|---|
| Concurrency | The 2025 benchmark ran a combination's scripts one after another | Requests run in parallel, each in its own process with its own model |
| Start | Scripts started whenever they were launched | All requests arrive together, then each loads its model and processes its folder |
| Request | Whole folder per script | Whole folder per request, plus every file logged |
| Times | Processing = mean time per file; execution = total time of the script incl. model load | Execution = processing + waiting (time without a free CPU) for the whole folder; model load in its own column |
| Threads | PyTorch defaults | CPU mode: the machine's threads split evenly between the requests (p > s > d for leftovers), passive waiting; GPU mode: defaults |
| Warm-up | None | None (cold runs) |
| Samples | 1 iteration | 5 iterations; every request and file recorded |
| Inputs | Different image folders on different machines | One fixed bundled set everywhere |
| System metrics | Read once after a script finished | Peaks and averages per file, request and run; raw ~0.1 s trace kept |
| Idle machine | Not checked | Checked before and during every run (incl. mains power); disturbed runs repeated |
| Sleep | Not handled | Blocked; interrupted runs detected and repeated |
| Setup | Installed by hand per machine, left behind | Detected per machine, PyTorch build tested on the GPU, removed automatically |
| Column names | `Image No`, `Processing Time`, `Execution Time (seconds)`, …; server CSVs `Total Processing Time (sec)`, … | The same names in `files.csv.gz` and `runs.csv`, plus raw columns. Run totals are those of the slowest request (the scripts ran one after another before, so they were sums) |
| Third task | Noise removal (2024) / speech (2025) | Speech, as in the earlier project code |

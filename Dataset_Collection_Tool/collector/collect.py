"""
Computation-latency collector.

A *run* = one combination (multiset of 1..5 requests from {s, d, p}) x one iteration, executed the
way an edge server serves concurrent requests:
  * every request is served independently by its own process with its own copy of the model
    ('ssp' -> two speech processes and one predict process);
  * a request = processing the task's whole input folder (all images / all audio clips);
  * all requests arrive at the same moment; each process then loads its model and processes its
    folder straight away - a cold run, no warm-up;
  * CPU mode: the machine's logical threads are split evenly between the requests of the run
    (leftover threads go to the slower tasks first: p > s > d), and idle OpenMP threads wait
    passively instead of spinning; GPU mode: PyTorch defaults.
Times of a request (none of them includes the model load, which is recorded separately):
  execution time  = wall-clock time to process the whole folder (first file -> last result)
  processing time = CPU time the request actually got while doing so
  waiting time    = execution time - processing time
Every request and every file is logged, so every task in a combination is a "target task".

Outputs (output/<machine_id>_<mode>/):
  files.csv.gz       one row per file processed (image / audio clip) - includes the column set of
                     the earlier datasets (Image No, Processing Time, Execution Time (seconds), ...)
  requests.csv       one row per request (one task's whole folder)
  runs.csv           one row per run - includes the columns of the earlier server CSVs
  monitor.csv.gz     raw 10 Hz system samples
  machine_info.json  hardware + software description
  collect.log        console log
This is raw data; conversion to a training format is done later, after the data of all machines
has been combined. Rerunning the same command resumes: runs marked 'ok' in runs.csv are skipped.
"""

import argparse
import bisect
import csv
import gzip
import itertools
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import psutil

from collector.config import FORMAT_VERSION, OUTPUT, ROOT, TASK_ORDER, TASKS, canonical, list_inputs
from collector.monitor import ForeignActivity, Gpu, Monitor, describe_busy, on_battery, swap_in_mb, system_activity
from collector.prompt import timed_input
from collector.sysinfo import machine_info
from collector.worker import worker_main

SCRIPT_NAMES = {"s": "speech.py", "d": "detect.py", "p": "predict.py"}   # "Current Script" of the earlier datasets
HEAVINESS = "psd"   # slowest task first (measured alone on the CPU with equal threads): leftover threads
                    # go to p, then s, then d

MONITOR_COLS = ["run_id", "t", "phase", "cpu_util", "cpu_per_core", "cpu_user_pct", "cpu_system_pct",
                "cpu_iowait_pct", "load_avg_1m", "ram_used_mb", "ram_used_pct", "swap_used_mb", "cpu_freq_mhz",
                "cpu_temp_c", "disk_read_mb_s", "disk_write_mb_s", "gpu_util", "gpu_mem_ctrl_util",
                "gpu_mem_used_mb", "gpu_clock_mhz", "gpu_temp_c", "gpu_power_w", "foreign_cpu_pct",
                "foreign_top", "foreign_gpu"]

STATIC_COLS = ["cpu_model", "cpu_physical_cores", "cpu_logical_cores", "cpu_max_freq_mhz",
               "ram_total_gb", "gpu_model", "gpu_mem_total_gb", "gpu_driver", "os", "python",
               "torch_version", "torch_cuda"]

# system state over a time window (a file, a request, a run), from the 100 ms monitor samples
PEAK_FIELDS = ["cpu_util", "cpu_iowait_pct", "load_avg_1m", "cpu_freq_mhz", "cpu_temp_c", "ram_used_mb",
               "ram_used_pct", "swap_used_mb", "gpu_util", "gpu_mem_used_mb", "gpu_clock_mhz", "gpu_temp_c",
               "gpu_power_w"]
AVG_FIELDS = ["cpu_util", "cpu_freq_mhz", "gpu_util", "gpu_clock_mhz"]
WINDOW_COLS = ["cpu_util_peak", "cpu_util_avg", "cpu_usage_per_core_peak", "cpu_usage_per_core_avg", "cores_used",
               "cpu_freq_mhz_peak", "cpu_freq_mhz_avg", "cpu_temp_c_peak", "cpu_iowait_pct_peak",
               "load_avg_1m_peak", "ram_used_mb_peak", "ram_used_pct_peak", "swap_used_mb_peak", "gpu_util_peak",
               "gpu_util_avg", "gpu_mem_used_mb_peak", "gpu_clock_mhz_peak", "gpu_clock_mhz_avg",
               "gpu_temp_c_peak", "gpu_power_w_peak", "n_monitor_samples"]

# ---- files.csv.gz: the column set of the earlier datasets (name -> where the value comes from) ----
FILE_LEGACY = {
    "Image No": "file_seq", "Image Pixel": "image_size", "Processing Time": "file_processing_ms",
    "Iteration": "iteration", "Combination": "combination", "Current Script": "script",
    "Execution Time (seconds)": "request_execution_time_s", "RAM Memory Usage (MB)": "process_rss_mb",
    "CPU Usage Per Core": "cpu_usage_per_core_peak", "GPU Usage (%)": "gpu_util_peak",
    "GPU Memory Usage (MB)": "process_gpu_mem_mb", "Total RAM (GB)": "ram_total_gb",
    "Total CPU Cores": "cpu_logical_cores", "Total GPU Memory (GB)": "gpu_mem_total_gb", "CPU Model": "cpu_model",
    "GPU Model": "gpu_model", "CPU Clock Speed (MHz)": "cpu_freq_mhz_avg", "GPU Clock Speed (MHz)": "gpu_clock_mhz_avg",
    "Number of Cores Used": "cores_used", "File": "input_file", "Duration": "audio_duration_s",
    "Number of Threads": "torch_threads", "Physical CPU Cores": "cpu_physical_cores",
    "Total RAM Used (MB)": "ram_used_mb_peak", "RAM Utilisation (%)": "ram_used_pct_peak",
}
FILE_COLS = [
    *FILE_LEGACY,
    "machine_id", "device_mode", "run_id", "concurrency", "task", "task_name", "slot",
    "input_bytes", "img_width", "img_height", "img_pixels", "audio_samples",
    "file_processing_ms", "file_execution_ms", "file_waiting_ms",
    "read_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "n_objects", "output_bytes", "requests_running",
    "cpu_util_peak", "cpu_util_avg", "cpu_freq_mhz_peak", "cpu_temp_c_peak", "cpu_iowait_pct_peak",
    "load_avg_1m_peak", "swap_used_mb_peak", "gpu_util_avg", "gpu_mem_used_mb_peak", "gpu_clock_mhz_peak",
    "gpu_temp_c_peak", "gpu_power_w_peak", "n_monitor_samples",
]

REQUEST_COLS = [
    "machine_id", "device_mode", "run_id", "combination", "concurrency", "n_s", "n_d", "n_p", "iteration",
    "task", "task_name", "script", "slot", "others_s", "others_d", "others_p",
    # times of the whole folder (model load not included) + the model load
    "execution_time_s", "processing_time_s", "waiting_time_s", "model_load_time_s", "running_at_end",
    # per file of this request
    "n_files", "file_execution_ms_mean", "file_execution_ms_p50", "file_execution_ms_p95", "file_execution_ms_p99",
    "file_execution_ms_max", "file_inference_ms_mean",
    # input / output
    "input_bytes_total", "img_pixels_total", "audio_duration_s_total", "first_img_pixels", "first_audio_samples",
    "n_objects_total", "output_bytes_total",
    # this request's process
    "threads_allocated", "torch_threads", "main_thread_cpu_time_s", "process_cpu_time_s", "threads_run_delay_s",
    "process_ctx_switches_involuntary",
    "process_peak_rss_mb", "process_gpu_peak_mb", "model_load_disk_read_mb",
    *WINDOW_COLS,
    *STATIC_COLS,
]

# ---- runs.csv: the columns of the earlier server CSVs (server1.csv GPU / server3.csv CPU) ----
RUN_LEGACY = {
    "Iteration": "iteration", "Combination": "combination",
    # run totals = the slowest request of the run (requests run in parallel; model load not included)
    "Total Processing Time (sec)": None, "Total Execution Time (sec)": None, "Total Waiting Time (sec)": None,
    "Scripts Executed": None,
    "Individual Processing Times": None, "Individual Execution Times": None,
    "Files Processed Per Script": None, "Num Files Processed": None,
    "First Image Pixels": None, "First Audio Samples": None,
    "Peak RAM Usage (MB)": "ram_used_mb_peak", "Peak CPU Usage (%)": "cpu_util_peak",
    "Peak GPU Usage (%)": "gpu_util_peak", "Peak GPU Memory (MB)": "gpu_mem_used_mb_peak",
    "Average CPU Usage Per Core": "cpu_usage_per_core_avg", "Number of Cores Used": "cores_used",
    "Average CPU Clock (MHz)": "cpu_freq_mhz_avg", "Average GPU Clock (MHz)": "gpu_clock_mhz_avg",
    "Total RAM (GB)": "ram_total_gb", "Total CPU Cores": "cpu_logical_cores",
    "Total GPU Memory (GB)": "gpu_mem_total_gb", "CPU Model": "cpu_model", "GPU Model": "gpu_model",
    "Device": None,
    # same style, added here
    "Individual Waiting Times": None, "Individual Model Load Times": None, "Threads Per Script": None,
}
RUN_COLS = [
    *RUN_LEGACY,
    "machine_id", "device_mode", "run_id", "concurrency", "n_s", "n_d", "n_p", "status", "error", "started_at",
    "pre_run_cpu_util", "pre_run_foreign_cpu", "idle_wait_s", "busy_processes", "foreign_cpu_avg",
    "foreign_cpu_max", "foreign_top_processes", "foreign_gpu_processes", "baseline_ram_used_mb",
    "baseline_gpu_mem_used_mb", "startup_s", "run_time_s", "requests", "suspended_s", "swap_in_mb",
    "cpu_util_avg", "cpu_usage_per_core_peak", "cpu_freq_mhz_peak", "cpu_temp_c_peak", "cpu_iowait_pct_peak",
    "load_avg_1m_peak", "ram_used_pct_peak", "swap_used_mb_peak", "gpu_util_avg", "gpu_clock_mhz_peak",
    "gpu_temp_c_peak", "gpu_power_w_peak",
    "duration_s",   # written last: a row with it filled was written completely
]
FILE_ALIAS = FILE_LEGACY
RUN_ALIAS = {k: v for k, v in RUN_LEGACY.items() if v}


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def __call__(self, msg=""):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e6 else f"{v:.3f}"
    return v


def append_rows(path, cols, rows, gz=False, alias=None):
    alias = alias or {}
    new = not path.exists()
    opener = gzip.open if gz else open
    with opener(path, "at" if gz else "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(alias.get(c, c))) for c in cols])


def read_table(path, gz=False):
    """(header, rows, damaged) - reads as much as possible of a file whose end may be cut off
    (machine switched off / killed while writing)."""
    if not path.exists():
        return None, [], False
    opener = gzip.open if gz else open
    header, rows, damaged = None, [], False
    try:
        with opener(path, "rt", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if header is None:
                    header = row
                elif len(row) == len(header):
                    rows.append(row)
                else:
                    damaged = True
    except (EOFError, OSError, zlib.error, csv.Error, UnicodeDecodeError):
        damaged = True
    return header, rows, damaged


def write_table(path, header, rows, gz=False):
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if gz else open
    with opener(tmp, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def repair_outputs(out, log):
    """Before resuming: drop what a run that did not finish left behind (rows of a run are written
    before the run's line in runs.csv, so rows of a run without a complete 'ok' line are partial),
    and cut off half-written lines. Afterwards every row belongs to exactly one completed run."""
    header, runs, damaged = read_table(out / "runs.csv")
    if header is None:
        return
    i_dur, i_id, i_st = header.index("duration_s"), header.index("run_id"), header.index("status")
    complete = [r for r in runs if r[i_dur] != ""]
    if damaged or len(complete) != len(runs):
        write_table(out / "runs.csv", header, complete)
        log("   repaired runs.csv (removed a line that was cut off)")
    ok = {r[i_id] for r in complete if r[i_st] == "ok"}
    for name, gz, filter_ok in (("requests.csv", False, True), ("files.csv.gz", True, True),
                                ("monitor.csv.gz", True, False)):
        h, rows, dmg = read_table(out / name, gz)
        if h is None:
            continue
        keep = [r for r in rows if r[h.index("run_id")] in ok] if filter_ok else rows
        if dmg or len(keep) != len(rows):
            write_table(out / name, h, keep, gz)
            log(f"   repaired {name}: removed {len(rows) - len(keep)} row(s) of an unfinished run"
                + (" and a line that was cut off" if dmg else ""))


class Samples:
    """Monitor samples with fast lookup of the samples covering a time window."""

    def __init__(self, samples):
        self.s = samples
        self.t = [x["t"] for x in samples]

    def covering(self, t0, t1):
        """A sample at time t reports usage over (previous sample, t], so the samples covering
        [t0, t1] are those inside it plus the first one after t1 (this also covers windows
        shorter than the sampling period)."""
        i = bisect.bisect_left(self.t, t0)
        j = bisect.bisect_right(self.t, t1)
        return self.s[i:min(j + 1, len(self.s))]


def _list(values):
    return "[" + ", ".join(f"{x:.1f}" for x in values) + "]"


def window_stats(samples):
    """Peaks and averages of the system state over the samples of a window."""
    out = {}
    for f in PEAK_FIELDS:
        vals = [s[f] for s in samples if s.get(f) is not None]
        out[f + "_peak"] = max(vals) if vals else None
    for f in AVG_FIELDS:
        vals = [s[f] for s in samples if s.get(f) is not None]
        out[f + "_avg"] = statistics.fmean(vals) if vals else None
    if samples:
        cores = list(zip(*(s["cpu_per_core"] for s in samples)))
        avg = [statistics.fmean(c) for c in cores]
        out["cpu_usage_per_core_peak"] = _list(max(c) for c in cores)
        out["cpu_usage_per_core_avg"] = _list(avg)
        out["cores_used"] = sum(1 for x in avg if x > 5.0)   # as in the earlier datasets: cores > 5 % busy
    else:
        out["cpu_usage_per_core_peak"] = out["cpu_usage_per_core_avg"] = out["cores_used"] = None
    out["n_monitor_samples"] = len(samples)
    return out


def suspend_clocks():
    """(monotonic, reference) - the monotonic clock stops while the machine is suspended; the
    reference keeps counting (Linux: CLOCK_BOOTTIME, elsewhere the wall clock)."""
    boot = getattr(time, "CLOCK_BOOTTIME", None)
    return time.monotonic(), (time.clock_gettime(boot) if boot is not None else time.time())


def suspended_seconds(clocks0):
    m0, r0 = clocks0
    m1, r1 = suspend_clocks()
    return max(0.0, (r1 - r0) - (m1 - m0))


def combinations(tasks, max_k):
    order = [t for t in TASK_ORDER if t in tasks]
    out = []
    for k in range(1, max_k + 1):
        out += ["".join(c) for c in itertools.combinations_with_replacement(order, k)]
    return out


def run_id_of(combo, it):
    return f"{combo}_it{it}"


# ------------------------------------------------------------------------------------------
# CPU threads per request
# ------------------------------------------------------------------------------------------
def cpu_thread_budget():
    """Logical CPU threads this process may use (respects CPU affinity / container limits)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1


def thread_plan(requests, budget):
    """Split `budget` threads evenly between the requests of a run; the leftover threads go one
    each to the slowest requests (p > s > d). 8 threads: s -> 8; spd -> p 3, s 3, d 2;
    sssss -> 2,2,2,1,1; ssppd -> p 2, p 2, s 2, s 1, d 1. At least 1 thread per request."""
    base, extra = divmod(budget, len(requests))
    if base == 0:
        base, extra = 1, 0
    order = sorted(requests, key=lambda r: (HEAVINESS.index(r[0]), r[1]))
    return {r: base + (1 if i < extra else 0) for i, r in enumerate(order)}


# ------------------------------------------------------------------------------------------
# making sure nothing else is running
# ------------------------------------------------------------------------------------------
def own_pids():
    """The collector, its worker processes, and the launcher chain above it (run.py and, when
    used, systemd-inhibit / caffeinate) - everything else counts as a foreign program."""
    me = psutil.Process()
    pids = {me.pid} | {c.pid for c in me.children(recursive=True)}
    p = me
    for _ in range(3):
        try:
            p = p.parent()
        except psutil.Error:
            break
        if p is None:
            break
        pids.add(p.pid)
    return pids


def leftover_processes():
    """Python processes started from this folder's environments that are not part of this
    collection: another collection still running, or request processes left behind by one that
    was killed. They would compete with the measurements."""
    venvs = [os.path.normcase(str(ROOT / v)) for v in (".venv-cpu", ".venv-gpu")]
    me = psutil.Process()
    mine = {me.pid} | {c.pid for c in me.children(recursive=True)} | {a.pid for a in me.parents()}
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
        except psutil.Error:
            continue
        if p.pid not in mine and cmd and any(os.path.normcase(cmd[0]).startswith(v) for v in venvs):
            found.append(p)
    return found


def stop_leftovers(log):
    procs = leftover_processes()
    if not procs:
        return
    log("Another collection from this folder (or processes left behind by one) is still running:")
    for p in procs:
        try:
            log(f"   PID {p.pid}: {' '.join(p.cmdline())[:150]}")
        except psutil.Error:
            pass
    try:
        ans = timed_input("   Stop them? Two collections at once would disturb each other's measurements. "
                          "[Y/n] (no answer in 2 min: yes) ", 120)
    except EOFError:
        ans = None
    if ans and ans.lower().startswith("n"):
        sys.exit("stopped: another collection is running on this machine.")
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=10)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    log(f"   stopped {len(procs)} process(es)")


def busy_reasons(args, a):
    """Why the machine is not ready for a measurement (empty list = idle and on mains power)."""
    reasons = []
    if a["cpu_util"] > args.idle_threshold:
        reasons.append(f"CPU {a['cpu_util']:.0f}% busy (limit {args.idle_threshold:.0f}%)")
    if a["foreign_cpu"] > args.foreign_threshold:
        reasons.append(f"other programs use {a['foreign_cpu']:.0f}% of the CPU (limit {args.foreign_threshold:.0f}%): "
                       + describe_busy(a, top=5))
    if a["gpu_procs"]:
        reasons.append("programs using the GPU: "
                       + "; ".join(f"{n}({pid}) {mb:.0f} MB" for n, pid, mb in a["gpu_procs"]))
    if a["gpu_util"] is not None and a["gpu_util"] > args.idle_threshold:
        reasons.append(f"GPU {a['gpu_util']}% busy")
    if a.get("on_battery") and not args.allow_battery:
        reasons.append("running on battery - plug the machine in (battery saving changes the timings)")
    return reasons


def preflight(args, gpu, log):
    """Before collection: check that no other program is using the CPU or the GPU."""
    if args.skip_idle_check:
        return
    for attempt in itertools.count(1):
        log("Checking that the machine is idle (5 s) ...")
        a = system_activity(gpu, 5.0, own_pids())
        problems = busy_reasons(args, a)
        if not problems:
            log(f"   idle: CPU {a['cpu_util']:.1f}%, other programs {a['foreign_cpu']:.1f}%"
                + (f", GPU {a['gpu_util']}%" if a["gpu_util"] is not None else ""))
            return
        log("   The machine is NOT ready:")
        for p in problems:
            log("     - " + p)
        log("   Close these programs (browser, IDE indexing, updates, other jobs) so they do not distort the data.")
        if attempt >= 6:
            log("   still not ready after 6 checks - continuing; every run is only measured when the machine "
                "is ready (other runs are skipped and retried)")
            return
        try:
            ans = timed_input("   [r]echeck now, [c]ontinue anyway, [q]uit?  (no answer: recheck in 5 min) ", 300)
        except EOFError:
            log("   (no keyboard) checking again in 60 s ...")
            time.sleep(60)
            continue
        if ans and ans.lower().startswith("c"):
            log("   continuing; every run is only measured when the machine is ready (others are skipped and retried)")
            return
        if ans and ans.lower().startswith("q"):
            sys.exit("stopped by user")


def wait_until_idle(args, gpu):
    """Before each run: wait (up to --idle-wait s) until the machine is idle: total CPU below
    --idle-threshold %, other programs below --foreign-threshold %, nothing else on the GPU,
    on mains power."""
    t = time.time()
    while True:
        a = system_activity(gpu, 1.0, own_pids())
        reasons = busy_reasons(args, a)
        if not reasons or time.time() - t >= args.idle_wait:
            return a, time.time() - t, reasons


# ------------------------------------------------------------------------------------------
# one run
# ------------------------------------------------------------------------------------------
def execute_run(combo, iteration, args, minfo, inputs, gpu, log):
    ctx = mp.get_context("spawn")
    counts = {t: combo.count(t) for t in "sdp"}
    plan = [(t, slot) for t in TASK_ORDER for slot in range(counts[t])]   # e.g. ssp -> s0 s1 p0
    threads = thread_plan(plan, args.cpu_threads) if args.mode == "cpu" else {r: None for r in plan}
    ready = ctx.Value("i", 0)
    go = ctx.Event()
    results = ctx.Queue()
    rid = run_id_of(combo, iteration)

    time.sleep(args.cooldown)
    base, idle_wait_s, busy_now = wait_until_idle(args, gpu)
    errors, done = {}, {}
    if busy_now:
        # never measure on a busy machine: skip the run (status failed_busy), it is retried later
        errors["*"] = {"message": f"machine not ready after {idle_wait_s:.0f} s: " + " | ".join(busy_now),
                       "busy": True}

    procs = {}
    for t, slot in (plan if not errors else []):
        spec = {
            "task": t, "slot": slot, "mode": args.mode, "threads": threads[(t, slot)],
            "files_per_request": args.files_per_request, "inputs": inputs[t],
            "seed": zlib.crc32(f"{args.seed}|{combo}|{iteration}|{t}{slot}".encode()),
        }
        p = ctx.Process(target=worker_main, args=(spec, ready, go, results), daemon=True)
        p.start()
        procs[(t, slot)] = p

    mon = Monitor(gpu, args.sample_interval, foreign=ForeignActivity(gpu, own_pids))
    mon.start()
    started_at = datetime.now().isoformat(timespec="seconds")
    t_launch = time.time()
    swap0 = swap_in_mb()

    def drain(block_s=0.0):
        try:
            while True:
                kind, key, payload = results.get(timeout=block_s) if block_s else results.get_nowait()
                (errors if kind == "error" else done)[key] = payload
                block_s = 0.0
        except queue.Empty:
            pass

    def dead():
        return [key for key, p in procs.items() if not p.is_alive() and key not in done and key not in errors]

    # ---- every request's process starts up (Python, PyTorch, model code; not timed) ----
    while not errors and ready.value < len(procs):
        drain()
        if errors or dead():
            break
        if time.time() - t_launch > args.ready_timeout:
            errors["*"] = {"message": f"request processes not ready after {args.ready_timeout:.0f}s", "oom": False}
            break
        time.sleep(0.05)
    startup_s = time.time() - t_launch

    # ---- t0: all requests arrive together; each loads its model and processes its folder ----
    t0 = None
    clocks0 = suspend_clocks()
    if not errors and not dead():
        mon.phase = "run"
        t0 = time.time()
        go.set()
        while len(done) + len(errors) < len(procs):
            drain(block_s=0.2)
            for key in dead():
                drain(block_s=1.0)
                if key not in done and key not in errors:
                    errors[key] = {"message": f"request process died (exit code {procs[key].exitcode}); "
                                              "likely killed by the OS for running out of memory", "oom": True}
            if time.time() - t0 > args.run_timeout:
                errors["*"] = {"message": f"run exceeded {args.run_timeout:.0f}s", "oom": False}
                break
    go.set()  # release anything still waiting
    for p in procs.values():
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
    samples = mon.stop()
    suspended_s = suspended_seconds(clocks0)
    swap1 = swap_in_mb()
    swap_in = (swap1 - swap0) if swap0 is not None and swap1 is not None else None

    # ---- other programs, sleep and power during the run ----
    run_samples = [s for s in samples if s["phase"] == "run"]
    f_vals = [s["foreign_cpu_pct"] for s in run_samples if s.get("foreign_cpu_pct") is not None]
    foreign_avg = statistics.fmean(f_vals) if f_vals else None
    foreign_max = max(f_vals) if f_vals else None
    top_count = {}
    for s in run_samples:
        for item in filter(None, (s.get("foreign_top") or "").split("; ")):
            name = item.split("(")[0]
            top_count[name] = top_count.get(name, 0) + 1
    foreign_top = "; ".join(n for n, _ in sorted(top_count.items(), key=lambda x: -x[1])[:5])
    foreign_gpu = "; ".join(sorted({g for s in run_samples for g in filter(None, (s.get("foreign_gpu") or "").split("; "))}))
    if not errors and ((foreign_avg or 0) > args.foreign_threshold or foreign_gpu):
        # another program competed with the measurement: discard the run, it is retried later
        why = (f"other programs used {foreign_avg:.1f}% of the CPU on average during the run "
               f"(limit {args.foreign_threshold:.0f}%): {foreign_top}" if (foreign_avg or 0) > args.foreign_threshold
               else f"other programs used the GPU during the run: {foreign_gpu}")
        errors["*"] = {"message": why, "busy": True}
    if not errors and on_battery() and not args.allow_battery:
        errors["*"] = {"message": "the machine switched to battery power during the run", "busy": True}
    if suspended_s > 2.0 and not errors:
        # the machine slept during the run: the timings include the sleep -> discard, redo later
        errors["*"] = {"message": f"machine was suspended for {suspended_s:.0f} s during the run", "oom": False,
                       "suspended": True}

    status, error_msg = "ok", ""
    if errors:
        e = next(iter(errors.values()))
        status = ("failed_busy" if any(v.get("busy") for v in errors.values())
                  else "failed_suspended" if any(v.get("suspended") for v in errors.values())
                  else "failed_oom" if any(v.get("oom") for v in errors.values()) else "failed")
        error_msg = " ".join(e["message"].split())[:300]
        log(f"   ! {status}: {error_msg}")
        if e.get("traceback") and args.verbose:
            log(e["traceback"])

    # ---- request and file rows ----
    run_s = Samples(run_samples)
    static = {c: minfo.get(c) for c in STATIC_COLS}
    ident = {"machine_id": args.machine_id, "device_mode": args.mode, "run_id": rid, "combination": combo,
             "concurrency": len(combo), "iteration": iteration, "n_s": counts["s"], "n_d": counts["d"],
             "n_p": counts["p"]}
    ordered = sorted(done.items(), key=lambda x: (TASK_ORDER.index(x[0][0]), x[0][1])) if status == "ok" else []
    reqs = [(t, slot, payload["request"]) for (t, slot), payload in ordered]
    ends = sorted(r["t_end"] for _, _, r in reqs)
    req_rows, file_rows = [], []
    for (t, slot), payload in ordered:
        r = payload["request"]
        row = {**ident, **r, **static}
        row.update({
            "task": t, "task_name": TASKS[t]["name"], "script": SCRIPT_NAMES[t], "slot": slot,
            "others_s": counts["s"] - (t == "s"), "others_d": counts["d"] - (t == "d"),
            "others_p": counts["p"] - (t == "p"), "threads_allocated": threads[(t, slot)],
            # requests of this run not yet finished when this one finished (incl. itself)
            "running_at_end": len(ends) - bisect.bisect_left(ends, r["t_end"]),
        })
        row.update(window_stats(run_s.covering(r["t_compute_start"], r["t_end"])))
        req_rows.append(row)

        for f in payload["files"]:
            frow = {**ident, **f, **static}
            frow.update({
                "task": t, "task_name": TASKS[t]["name"], "script": SCRIPT_NAMES[t],
                "torch_threads": r["torch_threads"], "request_execution_time_s": r["execution_time_s"],
                "image_size": f"{f['img_width']}x{f['img_height']}" if f.get("img_width") else None,
                # requests of this run not yet finished when this file finished (incl. its own)
                "requests_running": len(ends) - bisect.bisect_left(ends, f["t_end"]),
            })
            frow.update(window_stats(run_s.covering(f["t_start"], f["t_end"])))
            file_rows.append(frow)

    # ---- run summary (the columns of the earlier server CSVs + collection details) ----
    def joined(values, f="{:.3f}"):
        return ", ".join(f.format(v) for v in values) if reqs else None

    window = run_s.covering(min(r["t_compute_start"] for _, _, r in reqs), max(ends)) if reqs else run_s.s
    run_row = {**ident, **static, **window_stats(window)}
    # the requests run in parallel, so the run's execution time is that of its slowest request (the run is
    # done when it is done); processing / waiting time are those of the same request. No model load.
    slowest = max((r for _, _, r in reqs), key=lambda r: r["execution_time_s"]) if reqs else None
    run_row.update({
        "Total Processing Time (sec)": slowest["processing_time_s"] if slowest else None,
        "Total Execution Time (sec)": slowest["execution_time_s"] if slowest else None,
        "Total Waiting Time (sec)": slowest["waiting_time_s"] if slowest else None,
        "Scripts Executed": ", ".join(TASKS[t]["name"].capitalize() for t, _ in plan),
        "Individual Processing Times": joined(r["processing_time_s"] for _, _, r in reqs),
        "Individual Execution Times": joined(r["execution_time_s"] for _, _, r in reqs),
        "Individual Waiting Times": joined(r["waiting_time_s"] for _, _, r in reqs),
        "Individual Model Load Times": joined(r["model_load_time_s"] for _, _, r in reqs),
        "Threads Per Script": joined((r["torch_threads"] for _, _, r in reqs), "{}"),
        "Files Processed Per Script": joined((r["n_files"] for _, _, r in reqs), "{}"),
        "Num Files Processed": sum(r["n_files"] for _, _, r in reqs) if reqs else None,
        "First Image Pixels": next((r["first_img_pixels"] for t, _, r in reqs if t in "dp"), None),
        "First Audio Samples": next((r["first_audio_samples"] for t, _, r in reqs if t == "s"), None),
        "Device": args.mode.upper(),
        "status": status, "error": error_msg, "started_at": started_at,
        "pre_run_cpu_util": base["cpu_util"], "pre_run_foreign_cpu": base["foreign_cpu"],
        "idle_wait_s": idle_wait_s, "busy_processes": describe_busy(base),
        "foreign_cpu_avg": foreign_avg, "foreign_cpu_max": foreign_max,
        "foreign_top_processes": foreign_top, "foreign_gpu_processes": foreign_gpu,
        "baseline_ram_used_mb": base["ram_used_mb"], "baseline_gpu_mem_used_mb": base["gpu_mem_used_mb"],
        "startup_s": startup_s, "run_time_s": (max(ends) - t0) if reqs else None,
        "requests": len(req_rows), "suspended_s": suspended_s, "swap_in_mb": swap_in,
    })
    mon_rows = [dict(s, run_id=rid, cpu_per_core=";".join(f"{x:.0f}" for x in s["cpu_per_core"]))
                for s in samples]
    return run_row, req_rows, file_rows, mon_rows


# ------------------------------------------------------------------------------------------
# completeness check
# ------------------------------------------------------------------------------------------
def verify_output(out, plan, inputs, args, log):
    """Check that the dataset is complete and consistent: every planned run is ok (or did not fit in
    memory), every ok run has one row per request, every request has one row per file."""
    log("Checking the collected data ...")
    h, runs, _ = read_table(out / "runs.csv")
    status = {}
    for r in runs:
        status[r[h.index("run_id")]] = r[h.index("status")]
    planned = {run_id_of(c, it): c for c, it in plan}
    ok = {rid for rid in planned if status.get(rid) == "ok"}
    oom = sorted(rid for rid in planned if status.get(rid) == "failed_oom")
    missing = sorted(rid for rid in planned if rid not in ok and rid not in oom)

    def per_file_count(t):
        n = len(inputs[t])
        return n if args.files_per_request <= 0 else min(args.files_per_request, n)

    problems = []
    h, reqs, _ = read_table(out / "requests.csv")
    h = h or ["run_id"]
    req_n = {}
    for r in reqs:
        key = r[h.index("run_id")]
        req_n[key] = req_n.get(key, 0) + 1
    for rid in ok:
        if req_n.get(rid) != len(planned[rid]):
            problems.append(f"{rid}: {req_n.get(rid, 0)} request rows, expected {len(planned[rid])}")
    h, files, _ = read_table(out / "files.csv.gz", gz=True)
    h = h or ["run_id", "task", "slot"]
    file_n = {}
    i_run, i_task, i_slot = h.index("run_id"), h.index("task"), h.index("slot")
    for r in files:
        key = (r[i_run], r[i_task], r[i_slot])
        file_n[key] = file_n.get(key, 0) + 1
    for rid in ok:
        combo = planned[rid]
        for t in set(combo):
            for slot in range(combo.count(t)):
                got = file_n.get((rid, t, str(slot)), 0)
                if got != per_file_count(t):
                    problems.append(f"{rid} {t}{slot}: {got} file rows, expected {per_file_count(t)}")
    extra = {r[h.index("run_id")] for r in files} - ok
    if extra:
        problems.append(f"file rows of runs that are not ok: {sorted(extra)[:5]}")
    log(f"   runs: {len(ok)}/{len(plan)} ok" + (f", {len(oom)} did not fit in memory" if oom else "")
        + (f", {len(missing)} still failed: {', '.join(missing[:10])}" if missing else ""))
    log(f"   rows: {len(reqs)} requests, {len(files)} files" + ("" if problems else " - consistent"))
    for p in problems[:20]:
        log("   PROBLEM: " + p)
    return not missing and not problems


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Computation-latency collector",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--machine-id", required=True)
    ap.add_argument("--tasks", default="sdp", help="task letters to use")
    ap.add_argument("--max-concurrency", type=int, default=5, help="largest number of requests run together")
    ap.add_argument("--combos", help="comma-separated explicit combinations instead of all, e.g. s,dp,ssp")
    ap.add_argument("--iterations", type=int, default=5, help="repetitions of every combination")
    ap.add_argument("--files-per-request", type=int, default=0,
                    help="files processed by one request: 0 = the whole input folder (default), "
                         "N = a seeded random subset of N files")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="CPU mode: threads split between the requests of a run (0 = all logical threads)")
    ap.add_argument("--idle-threshold", type=float, default=10.0,
                    help="CPU %% above which the machine counts as busy")
    ap.add_argument("--foreign-threshold", type=float, default=5.0,
                    help="max CPU %% (of the whole machine) that other programs may use before / during a run")
    ap.add_argument("--idle-wait", type=float, default=120.0,
                    help="max wait for the machine to become ready before a run (s); still busy -> run skipped")
    ap.add_argument("--retry-passes", type=int, default=2,
                    help="extra passes over runs that failed (busy, suspended, errors) after the main pass")
    ap.add_argument("--skip-idle-check", action="store_true", help="skip the idle check before collection")
    ap.add_argument("--allow-battery", action="store_true", help="measure even when a laptop runs on battery")
    ap.add_argument("--cooldown", type=float, default=3.0, help="pause between runs (s)")
    ap.add_argument("--sample-interval", type=float, default=0.1, help="system monitor period (s)")
    ap.add_argument("--ready-timeout", type=float, default=900.0, help="max time for the request processes to start (s)")
    ap.add_argument("--run-timeout", type=float, default=7200.0, help="max time for one run (s)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out-dir", help="output folder (default output/<machine_id>_<mode>)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick check: 5 short runs (20 files per request) into a separate folder")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--verbose", action="store_true", help="print worker tracebacks")
    args = ap.parse_args()
    if args.smoke:
        args.combos = args.combos or "s,d,p,sdp,ssppd"
        args.iterations, args.cooldown, args.files_per_request = 1, 1.0, args.files_per_request or 20
    if args.cpu_threads <= 0:
        args.cpu_threads = cpu_thread_budget()
    return args


def isolate_caches():
    """Keep every library cache / settings file inside the environment folder, also when the
    collector is started directly instead of through run.py (which sets the same variables)."""
    cache = Path(sys.prefix) / "cache"
    for var, sub in (("XDG_CACHE_HOME", ""), ("XDG_CONFIG_HOME", "config"), ("HF_HOME", "huggingface"),
                     ("TORCH_HOME", "torch"), ("MPLCONFIGDIR", "matplotlib"), ("YOLO_CONFIG_DIR", "ultralytics"),
                     ("CUDA_CACHE_PATH", "nv")):
        os.environ.setdefault(var, str(cache / sub) if sub else str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    isolate_caches()
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    args = parse_args()
    tasks = [t for t in args.tasks if t in TASKS]
    if args.combos:
        combos = sorted({canonical(c.strip()) for c in args.combos.split(",") if c.strip()},
                        key=lambda c: (len(c), c))
        bad = [c for c in combos if any(t not in TASKS for t in c)]
        if bad:
            sys.exit(f"unknown task letters in {bad}; valid letters: {''.join(TASKS)}")
        tasks = sorted({t for c in combos for t in c}, key=TASK_ORDER.index)
    else:
        combos = combinations(tasks, args.max_concurrency)

    plan = []
    for it in range(1, args.iterations + 1):  # iteration-major, shuffled -> repetitions spread over time
        order = combos[:]
        random.Random(args.seed + it).shuffle(order)
        plan += [(c, it) for c in order]

    name = f"{args.machine_id}_{args.mode}" + ("_smoke" if args.smoke else "")
    out = Path(args.out_dir) if args.out_dir else OUTPUT / name
    thread_note = (f"even split of {args.cpu_threads} logical threads between the requests of a run "
                   f"(leftover threads to {' > '.join(HEAVINESS)}); OMP_WAIT_POLICY=PASSIVE" if args.mode == "cpu"
                   else "PyTorch defaults (GPU mode)")

    print(f"Combinations: {len(combos)} (concurrency 1..{max(map(len, combos))}), iterations: {args.iterations}, "
          f"runs: {len(plan)}\nOutput folder: {out}\nThreads: {thread_note}")
    if args.dry_run:
        for k in range(1, max(map(len, combos)) + 1):
            print(f"  k={k}: {' '.join(c for c in combos if len(c) == k)}")
        if args.mode == "cpu":
            for c in ("s", "spd", "sssss", "ssppd"):
                tp = thread_plan([(t, i) for t in TASK_ORDER for i in range(c.count(t))], args.cpu_threads)
                print(f"  threads for {c}: " + " ".join(f"{t}{i}:{n}" for (t, i), n in tp.items()))
        return

    out.mkdir(parents=True, exist_ok=True)
    log = Tee(out / "collect.log")
    stop_leftovers(log)
    gpu = Gpu(enabled=(args.mode == "gpu"))  # CPU mode: GPU columns stay empty

    info_path = out / "machine_info.json"
    minfo = machine_info(args.machine_id, args.mode, Gpu())
    minfo.update({"format_version": FORMAT_VERSION, "thread_settings": thread_note,
                  "waiting_time_method": ("Linux scheduler run delay of the main thread (schedstat)"
                                          if os.path.exists("/proc/thread-self/schedstat")
                                          else "execution time - main-thread CPU time"),
                  "files_per_request": args.files_per_request, "warmup": "none (cold runs)"})
    if info_path.exists():
        old = json.loads(info_path.read_text())
        for key in ("format_version", "cpu_model", "gpu_model", "device_mode", "inputs_fingerprint",
                    "files_per_request", "warmup", "thread_settings"):
            if old.get(key) != minfo.get(key):
                sys.exit(f"{out} was created with {key}={old.get(key)!r} but now {key}={minfo.get(key)!r}. "
                         "Keep settings identical within a dataset, or use a different --machine-id / --out-dir.")
    else:
        info_path.write_text(json.dumps(minfo, indent=2))
    if args.mode == "gpu" and not minfo.get("cuda_available"):
        sys.exit("GPU mode selected but torch.cuda.is_available() is False.")

    inputs = {t: list_inputs(t) for t in tasks}
    for t, files in inputs.items():
        if not files:
            sys.exit(f"no input files found for task '{t}' in {TASKS[t]['input_dir']}")

    runs_path = out / "runs.csv"
    repair_outputs(out, log)

    def statuses_now():
        h, rows, _ = read_table(runs_path)
        return {r[h.index("run_id")]: r[h.index("status")] for r in rows} if h else {}

    finished = {rid for rid, st in statuses_now().items() if st == "ok"}
    todo = [(c, it) for c, it in plan if run_id_of(c, it) not in finished]

    log("=" * 70)
    log(f"machine={args.machine_id} mode={args.mode} cpu={minfo['cpu_model']} "
        f"gpu={minfo.get('gpu_model')} torch={minfo.get('torch_version')}")
    log(f"cores: {minfo.get('cpu_physical_cores')} physical / {minfo.get('cpu_logical_cores')} logical | "
        f"threads: {thread_note}")
    log("request = " + ("whole input folder (" + ", ".join(f"{t}: {len(v)} files" for t, v in inputs.items()) + ")"
                        if args.files_per_request <= 0 else f"{args.files_per_request} files")
        + " | each request: own process + own model, all arrive together, cold run (no warm-up)")
    log(f"{len(plan)} runs planned, {len(plan) - len(todo)} already done, {len(todo)} to go")
    if todo:
        preflight(args, gpu, log)

    t_start, durations = time.time(), []
    work = [(combo, it, 0) for combo, it in todo]
    n = 0
    while work:
        combo, it, pass_no = work.pop(0)
        n += 1
        t_run = time.time()
        label = f"[{n}/{n + len(work)}]" + (f" retry pass {pass_no}" if pass_no else "")
        log(f"{label} run {run_id_of(combo, it)}  (k={len(combo)})")
        try:
            run_row, req_rows, file_rows, mon_rows = execute_run(combo, it, args, minfo, inputs, gpu, log)
        except KeyboardInterrupt:
            log("Interrupted - partial run discarded. Rerun the same command to resume.")
            sys.exit(130)
        run_row["duration_s"] = time.time() - t_run
        append_rows(out / "requests.csv", REQUEST_COLS, req_rows)
        append_rows(out / "files.csv.gz", FILE_COLS, file_rows, gz=True, alias=FILE_ALIAS)
        append_rows(out / "monitor.csv.gz", MONITOR_COLS, mon_rows, gz=True)
        append_rows(runs_path, RUN_COLS, [run_row], alias=RUN_ALIAS)  # written last: marks the run as complete

        durations.append(run_row["duration_s"])
        eta = statistics.fmean(durations) * len(work)
        if req_rows:
            per = "  ".join(f"{r['task']}{r['slot']}[{r['torch_threads']}t] load {r['model_load_time_s']:.1f} "
                            f"exec {r['execution_time_s']:.1f} proc {r['processing_time_s']:.1f}" for r in req_rows)
            log(f"   {run_row['status']}  (s)  {per}  | run {run_row['run_time_s']:.0f} s, "
                f"cpu peak {run_row['cpu_util_peak'] or 0:.0f}%, others {run_row['foreign_cpu_avg'] or 0:.1f}%  | "
                f"ETA {eta / 3600:.1f} h")
        else:
            log(f"   {run_row['status']}  | ETA {eta / 3600:.1f} h")
        if not work and pass_no < args.retry_passes:
            # end of a pass: queue the runs that failed for a reason other than memory, once more
            st = statuses_now()
            retry = [(c, i, pass_no + 1) for c, i in plan if st.get(run_id_of(c, i)) not in ("ok", "failed_oom")]
            if retry:
                log(f"Retry pass {pass_no + 1}: {len(retry)} run(s) failed (busy / suspended / error) - running them again")
                work = retry

    log(f"Finished in {(time.time() - t_start) / 3600:.2f} h. Data in {out}")
    complete = verify_output(out, plan, inputs, args, log)
    if not complete:
        log("Some runs are missing or failed (see runs.csv). Rerun the same command to retry them.")
    # exit code 0 when the dataset is complete (every run ok or failed_oom), 3 otherwise
    sys.exit(0 if complete else 3)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()

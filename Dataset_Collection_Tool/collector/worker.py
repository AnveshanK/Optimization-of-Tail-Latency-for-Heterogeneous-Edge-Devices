"""
One worker process = one request of a run, served independently like a request on an edge server
(in `ssp` there are three worker processes, each with its own copy of its model).

Lifecycle of a worker in a run (a cold run):
  1. start up: Python, PyTorch, the model code and (GPU mode) the CUDA context; read the input
     features of the folder                                             - not timed
  2. report "ready" and wait until every request of the run is ready
  3. GO (all requests of the run arrive together):
       load the model                                                   - timed: model load time
       process every file of the folder one after another, cold (no warm-up)
Times of the folder (none of them includes the model load):
  execution time  = wall-clock time from the first file's start to the last file's end
  waiting time    = time the request's main thread was ready to run but had no free CPU core
                    (Linux scheduler "run delay", /proc/thread-self/schedstat). The main thread
                    takes part in every computation of the request, so while it waits the
                    request is stalled. Other systems: execution time - main-thread CPU time
  processing time = execution time - waiting time
The same three times are also recorded for every file.

Threads: in CPU mode the collector gives every request a number of threads (an even share of the
machine's logical threads), and idle OpenMP threads wait passively; in GPU mode PyTorch's defaults
are used.

Nothing heavy is imported at module level: with the 'spawn' start method the child imports this
module first, and CUDA_VISIBLE_DEVICES / OMP_NUM_THREADS must be set before torch is imported.
"""

import os
import random
import statistics
import time
import traceback

def _pct(values, q):
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _io_read_mb(proc):
    try:
        return proc.io_counters().read_bytes / 2**20   # bytes this process read from disk
    except Exception:  # noqa: BLE001 - not available on every OS
        return None


def _peak_rss_mb():
    try:
        import resource
        import sys
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 2**20 if sys.platform == "darwin" else kb / 2**10
    except Exception:  # noqa: BLE001 - Windows
        return None


RUN_DELAY_FILE = "/proc/thread-self/schedstat"   # "<on-CPU ns> <run-delay ns> <timeslices>" (Linux)


def run_delay_s():
    """Run delay of the calling thread so far (s), or None where the OS does not report it."""
    try:
        with open(RUN_DELAY_FILE) as f:
            return int(f.read().split()[1]) / 1e9
    except (OSError, ValueError, IndexError):
        return None


def all_threads_run_delay_s():
    """Run delay summed over all threads of this process (s), or None."""
    total, found = 0, False
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        return None
    for tid in tids:
        try:
            with open(f"/proc/self/task/{tid}/schedstat") as f:
                total += int(f.read().split()[1])
                found = True
        except (OSError, ValueError, IndexError):
            pass
    return total / 1e9 if found else None


def split_times(exec_s, delay0, delay1, cpu_s):
    """(processing, waiting) for an interval of `exec_s` wall-clock seconds."""
    if delay0 is not None and delay1 is not None:
        wait = min(exec_s, max(0.0, delay1 - delay0))
    else:
        wait = max(0.0, exec_s - cpu_s)
    return exec_s - wait, wait


def worker_main(spec, ready_count, go_event, result_queue):
    task_letter, slot = spec["task"], spec["slot"]
    try:
        if spec["mode"] == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if spec["threads"]:
            # thread count given by the collector (CPU mode): set before torch is imported so
            # the OpenMP / MKL thread pools are created with this size
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[var] = str(spec["threads"])
            # idle OpenMP threads sleep instead of spinning: spinning threads take CPU time from the
            # working threads (hyper-thread siblings) and from the other requests of the run
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("YOLO_OFFLINE", "1")
        os.environ.setdefault("YOLO_VERBOSE", "False")
        if os.environ.get("YOLO_CONFIG_DIR"):
            # own Ultralytics settings folder per process: requests start at the same moment and
            # would otherwise write the same settings file
            os.environ["YOLO_CONFIG_DIR"] = os.path.join(os.environ["YOLO_CONFIG_DIR"], f"{task_letter}{slot}")

        import psutil
        import torch

        from collector.tasks import TASK_CLASSES, feature_reader, prepare_imports

        device = torch.device("cuda:0" if spec["mode"] == "gpu" else "cpu")
        prepare_imports(task_letter)
        import logging
        logging.getLogger("yolov5").setLevel(logging.ERROR)   # no "Fusing layers..." / model summary lines
        if spec["threads"]:
            torch.set_num_threads(spec["threads"])   # after the model code is imported (it touches thread settings)
        if device.type == "cuda":
            torch.zeros(1, device=device)         # CUDA context exists before anything is timed
            torch.cuda.synchronize()
        proc = psutil.Process()

        # the folder this request processes (whole folder by default; a seeded subset if requested)
        inputs = spec["inputs"]
        n = spec["files_per_request"]
        folder = sorted(random.Random(spec["seed"]).sample(inputs, n)) if 0 < n < len(inputs) else list(inputs)
        read_feat = feature_reader(task_letter)
        features = {p: read_feat(p) for p in folder}

        # ---- wait until every request of the run is ready; then all arrive together ----
        with ready_count.get_lock():
            ready_count.value += 1
        go_event.wait()

        # ---- model load (timed separately; not part of execution / processing time) ----
        io0 = _io_read_mb(proc)
        p0 = time.perf_counter()
        task = TASK_CLASSES[task_letter](device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_load_ms = (time.perf_counter() - p0) * 1000.0
        io1 = _io_read_mb(proc)

        # ---- the folder, cold ----
        rows = []
        cpu_a, ctx_a = proc.cpu_times(), proc.num_ctx_switches()
        all_delay_a = all_threads_run_delay_s()
        t_start = time.time()
        p_start, c_start, d_start = time.perf_counter(), time.thread_time(), run_delay_s()
        for i, path in enumerate(folder, 1):
            w0, p_f, c_f, d_f = time.time(), time.perf_counter(), time.thread_time(), run_delay_s()
            res = task.process(path)
            exec_s = time.perf_counter() - p_f
            proc_s, wait_s = split_times(exec_s, d_f, run_delay_s(), time.thread_time() - c_f)
            row = {"slot": slot, "file_seq": i, "input_file": os.path.basename(path),
                   "t_start": w0, "t_end": time.time(),
                   "file_execution_ms": exec_s * 1000.0, "file_processing_ms": proc_s * 1000.0,
                   "file_waiting_ms": wait_s * 1000.0,
                   "process_rss_mb": proc.memory_info().rss / 2**20,
                   "process_gpu_mem_mb": (torch.cuda.memory_reserved() / 2**20 if device.type == "cuda" else None)}
            row.update(features[path])
            row.update(res)
            rows.append(row)
        execution_s = time.perf_counter() - p_start
        main_cpu_s = time.thread_time() - c_start
        processing_s, waiting_s = split_times(execution_s, d_start, run_delay_s(), main_cpu_s)
        t_end = time.time()
        cpu_b, ctx_b = proc.cpu_times(), proc.num_ctx_switches()
        all_delay_b = all_threads_run_delay_s()

        per_file = [r["file_execution_ms"] for r in rows]
        total = (lambda key: sum(features[p].get(key) or 0 for p in folder))
        first = features[folder[0]]
        request = {
            "slot": slot, "t_compute_start": t_start, "t_end": t_end,
            "execution_time_s": execution_s,
            "processing_time_s": processing_s,
            "waiting_time_s": waiting_s,
            "model_load_time_s": model_load_ms / 1000.0,
            "main_thread_cpu_time_s": main_cpu_s,
            "threads_run_delay_s": (all_delay_b - all_delay_a)
            if all_delay_a is not None and all_delay_b is not None else None,
            "model_load_disk_read_mb": (io1 - io0) if io0 is not None and io1 is not None else None,
            "n_files": len(rows),
            "file_execution_ms_mean": statistics.fmean(per_file),
            "file_execution_ms_p50": _pct(per_file, 0.5),
            "file_execution_ms_p95": _pct(per_file, 0.95),
            "file_execution_ms_p99": _pct(per_file, 0.99),
            "file_execution_ms_max": max(per_file),
            "file_inference_ms_mean": statistics.fmean(r["inference_ms"] for r in rows),
            "input_bytes_total": total("input_bytes"),
            "img_pixels_total": total("img_pixels") or None,
            "audio_duration_s_total": total("audio_duration_s") or None,
            "first_img_pixels": first.get("img_pixels"),
            "first_audio_samples": first.get("audio_samples"),
            "n_objects_total": sum(r["n_objects"] for r in rows),
            "output_bytes_total": sum(r["output_bytes"] for r in rows),
            "torch_threads": torch.get_num_threads(),
            # CPU time of ALL threads of the process (can exceed the execution time with several threads)
            "process_cpu_time_s": (cpu_b.user - cpu_a.user) + (cpu_b.system - cpu_a.system),
            "process_ctx_switches_involuntary": ctx_b.involuntary - ctx_a.involuntary,
            "process_peak_rss_mb": _peak_rss_mb() or proc.memory_info().rss / 2**20,
            "process_gpu_peak_mb": (torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None),
        }
        result_queue.put(("done", (task_letter, slot), {"request": request, "files": rows}))
    except BaseException as e:  # noqa: BLE001 - report everything (incl. CUDA OOM) to the main process
        msg = f"{type(e).__name__}: {e}"
        oom = "out of memory" in str(e).lower() or isinstance(e, MemoryError)
        result_queue.put(("error", (task_letter, slot), {"message": msg, "oom": oom, "traceback": traceback.format_exc()}))

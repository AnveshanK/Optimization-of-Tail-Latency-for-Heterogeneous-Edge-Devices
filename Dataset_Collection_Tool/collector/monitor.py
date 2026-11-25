"""Background sampler for system-wide CPU / RAM / GPU state (runs in the main process)."""

import threading
import time

import psutil

try:
    import pynvml
except Exception:  # noqa: BLE001
    pynvml = None


class Gpu:
    """Thin NVML wrapper; every method returns None when no NVIDIA GPU is usable."""

    def __init__(self, index=0, enabled=True):
        self.handle = None
        if pynvml is None or not enabled:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001
            self.handle = None

    def _try(self, fn, *a):
        if self.handle is None:
            return None
        try:
            return fn(self.handle, *a)
        except Exception:  # noqa: BLE001
            return None

    def sample(self):
        util = self._try(pynvml.nvmlDeviceGetUtilizationRates) if self.handle else None
        mem = self._try(pynvml.nvmlDeviceGetMemoryInfo) if self.handle else None
        power = self._try(pynvml.nvmlDeviceGetPowerUsage) if self.handle else None
        return {
            "gpu_util": util.gpu if util else None,
            "gpu_mem_ctrl_util": util.memory if util else None,  # % time the GPU memory was read/written
            "gpu_mem_used_mb": mem.used / 2**20 if mem else None,
            "gpu_clock_mhz": self._try(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM) if self.handle else None,
            "gpu_temp_c": self._try(pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU) if self.handle else None,
            "gpu_power_w": power / 1000.0 if power is not None else None,
        }

    def static_info(self):
        if self.handle is None:
            return {}
        name = self._try(pynvml.nvmlDeviceGetName)
        mem = self._try(pynvml.nvmlDeviceGetMemoryInfo)
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
        except Exception:  # noqa: BLE001
            driver = None
        dec = (lambda v: v.decode() if isinstance(v, bytes) else v)
        return {
            "gpu_model": dec(name),
            "gpu_mem_total_gb": mem.total / 2**30 if mem else None,
            "gpu_driver": dec(driver),
            "gpu_max_sm_clock_mhz": self._try(pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_SM),
            "gpu_count": pynvml.nvmlDeviceGetCount(),
        }


def cpu_temp_c():
    """Hottest CPU temperature sensor (Linux; None where not available)."""
    try:
        temps = psutil.sensors_temperatures()
    except Exception:  # noqa: BLE001
        return None
    vals = [t.current for name, entries in (temps or {}).items()
            if name in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz") for t in entries if t.current]
    return max(vals) if vals else None


def load_avg_1m():
    try:
        return psutil.getloadavg()[0]
    except Exception:  # noqa: BLE001
        return None


def on_battery():
    """True when a laptop runs on battery (battery saving changes CPU / GPU speed), False on mains
    power, None when unknown. Desktops / servers without a battery count as mains power."""
    try:
        b = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        return None
    if b is None:
        return False
    return None if b.power_plugged is None else not b.power_plugged


def swap_in_mb():
    """Total data read back from swap since boot (MB): growth during a run means RAM ran out."""
    try:
        return psutil.swap_memory().sin / 2**20
    except Exception:  # noqa: BLE001
        return None


def cpu_freq_mhz():
    try:
        f = psutil.cpu_freq()
        return f.current if f else None
    except Exception:  # noqa: BLE001
        return None


class ForeignActivity:
    """CPU used by programs other than the collector, and other programs computing on the GPU.
    `own_pids` is a callable returning the collector's own process ids at that moment."""

    def __init__(self, gpu: Gpu, own_pids):
        self.gpu, self.own_pids = gpu, own_pids
        self.n_cpu = psutil.cpu_count() or 1
        self.measure()  # prime the per-process CPU counters

    def measure(self):
        own = self.own_pids()
        total, top = 0.0, []
        for p in psutil.process_iter(["pid", "name"]):
            if p.info["pid"] in own:
                continue
            try:
                share = p.cpu_percent(None) / self.n_cpu   # % of the whole machine since last call
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            total += share
            if share >= 1.0:
                top.append((share, p.info["name"], p.info["pid"]))
        top.sort(reverse=True)
        gpu_procs = []
        if self.gpu.handle is not None:
            try:
                for gp in pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu.handle):
                    if gp.pid not in own:
                        try:
                            gpu_procs.append(psutil.Process(gp.pid).name())
                        except psutil.Error:
                            gpu_procs.append(str(gp.pid))
            except Exception:  # noqa: BLE001
                pass
        return {"foreign_cpu_pct": total,
                "foreign_top": "; ".join(f"{n}({pid}) {sh:.0f}%" for sh, n, pid in top[:3]),
                "foreign_gpu": "; ".join(gpu_procs)}


class Monitor:
    def __init__(self, gpu: Gpu, interval=0.1, foreign: "ForeignActivity" = None, foreign_period=1.0):
        self.gpu, self.interval = gpu, interval
        self.samples, self._stop, self._thread = [], threading.Event(), None
        self.phase = "load"
        self._disk, self._disk_t = None, None
        # other programs are measured once per second (listing all processes is too costly at 10 Hz);
        # every 100 ms sample carries the latest measurement
        self.foreign, self.foreign_period = foreign, foreign_period
        self._foreign_last, self._foreign_t = {"foreign_cpu_pct": None, "foreign_top": "", "foreign_gpu": ""}, 0.0

    def _disk_rates(self, now):
        """Disk read / write speed (MB/s) since the previous sample."""
        try:
            d = psutil.disk_io_counters()
        except Exception:  # noqa: BLE001
            d = None
        rates = (None, None)
        if d is not None and self._disk is not None and now > self._disk_t:
            dt = now - self._disk_t
            rates = ((d.read_bytes - self._disk.read_bytes) / 2**20 / dt,
                     (d.write_bytes - self._disk.write_bytes) / 2**20 / dt)
        self._disk, self._disk_t = d, now
        return rates

    def _one(self):
        now = time.time()
        per_core = psutil.cpu_percent(percpu=True)
        split = psutil.cpu_times_percent(interval=None)
        vm = psutil.virtual_memory()
        read_mb_s, write_mb_s = self._disk_rates(now)
        s = {
            "t": now,
            "phase": self.phase,
            "cpu_util": sum(per_core) / len(per_core),
            "cpu_per_core": per_core,
            "cpu_user_pct": split.user,
            "cpu_system_pct": split.system,
            "cpu_iowait_pct": getattr(split, "iowait", None),   # waiting for disk (Linux)
            "load_avg_1m": load_avg_1m(),
            "ram_used_mb": (vm.total - vm.available) / 2**20,
            "ram_used_pct": vm.percent,
            "swap_used_mb": psutil.swap_memory().used / 2**20,
            "cpu_freq_mhz": cpu_freq_mhz(),
            "cpu_temp_c": cpu_temp_c(),
            "disk_read_mb_s": read_mb_s,
            "disk_write_mb_s": write_mb_s,
        }
        s.update(self.gpu.sample())
        if self.foreign is not None and now - self._foreign_t >= self.foreign_period:
            self._foreign_last, self._foreign_t = self.foreign.measure(), now
        s.update(self._foreign_last)
        return s

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._one())
            self._stop.wait(self.interval)

    def start(self):
        psutil.cpu_percent(percpu=True)  # prime the counters
        psutil.cpu_times_percent(interval=None)
        self._disk_rates(time.time())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples


def system_activity(gpu: Gpu, seconds=1.0, exclude_pids=()):
    """Measure the machine over `seconds`: total CPU %, RAM, GPU state, and the other processes
    that are using CPU (% of the whole machine) or the GPU. Used to make sure nothing else is
    running before collection and before every run."""
    exclude = set(exclude_pids)
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        if p.info["pid"] in exclude:
            continue
        try:
            p.cpu_percent(None)
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.cpu_percent(percpu=True)
    time.sleep(seconds)
    per_core = psutil.cpu_percent(percpu=True)
    n_cpu = psutil.cpu_count() or 1
    busy, foreign = [], 0.0
    for p in procs:
        try:
            share = p.cpu_percent(None) / n_cpu   # % of the whole machine
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        foreign += share
        if share >= 2.0:
            busy.append((share, p.info["name"], p.info["pid"]))
    busy.sort(reverse=True)
    vm = psutil.virtual_memory()
    g = gpu.sample()
    gpu_procs = []
    if gpu.handle is not None:
        try:
            for gp in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu.handle):
                if gp.pid in exclude:
                    continue
                try:
                    name = psutil.Process(gp.pid).name()
                except psutil.Error:
                    name = "?"
                gpu_procs.append((name, gp.pid, (gp.usedGpuMemory or 0) / 2**20))
        except Exception:  # noqa: BLE001
            pass
    return {
        "cpu_util": sum(per_core) / len(per_core),
        "ram_used_mb": (vm.total - vm.available) / 2**20,
        "gpu_util": g["gpu_util"],
        "gpu_mem_used_mb": g["gpu_mem_used_mb"],
        "on_battery": on_battery(),
        "foreign_cpu": foreign,    # CPU used by all other programs together (% of the machine)
        "busy": busy,              # [(% of machine, name, pid)]
        "gpu_procs": gpu_procs,    # [(name, pid, MB)] other programs computing on the GPU
    }


def describe_busy(activity, top=3):
    return "; ".join(f"{name}({pid}) {share:.0f}%" for share, name, pid in activity["busy"][:top])

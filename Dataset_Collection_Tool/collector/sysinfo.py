"""Static description of the machine and software stack, stored with every dataset."""

import hashlib
import os
import platform
import socket
import subprocess
import sys

import psutil

from collector.config import TASKS, list_inputs


def cpu_model():
    try:
        import cpuinfo  # py-cpuinfo, installed with ultralytics
        name = cpuinfo.get_cpu_info().get("brand_raw")
        if name:
            return name
    except Exception:  # noqa: BLE001
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def inputs_fingerprint():
    """Hash of the input set, so we can verify all machines used identical inputs."""
    h = hashlib.sha256()
    for t in sorted(TASKS):
        for p in list_inputs(t):
            h.update(f"{t}|{os.path.basename(p)}|{os.path.getsize(p)}".encode())
    return h.hexdigest()[:16]


def torch_info(mode):
    """Query torch in a throw-away subprocess so the main process never initialises CUDA."""
    code = (
        "import json, torch, transformers;"
        "d={'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cudnn':(torch.backends.cudnn.version() if torch.cuda.is_available() else None),"
        "'cuda_available':torch.cuda.is_available(),'torch_default_threads':torch.get_num_threads(),"
        "'transformers_version':transformers.__version__};print(json.dumps(d))"
    )
    env = dict(os.environ)
    if mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    import json
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {}


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def power_info():
    """Power settings that change how fast the machine computes (recorded, not changed)."""
    from collector.monitor import on_battery
    info = {"on_battery_at_start": on_battery()}
    if sys.platform.startswith("linux"):
        info["cpu_governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        info["cpu_energy_preference"] = _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
        try:
            r = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=10)
            info["power_profile"] = r.stdout.strip() or None if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            info["power_profile"] = None
    elif os.name == "nt":
        try:
            r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
            info["power_profile"] = r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            info["power_profile"] = None
    return info


def machine_info(machine_id, mode, gpu):
    freq = psutil.cpu_freq()
    info = {
        "machine_id": machine_id,
        "device_mode": mode,
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_freq_mhz": freq.max if freq and freq.max else None,
        "ram_total_gb": psutil.virtual_memory().total / 2**30,
        "gpu_model": None,
        "gpu_mem_total_gb": None,
        "inputs_fingerprint": inputs_fingerprint(),
    }
    info.update(power_info())
    info.update(gpu.static_info())
    info.update(torch_info(mode))
    if mode == "cpu":
        # the GPU (if any) is not used in CPU mode; keep it recorded but clearly separate
        info["gpu_present_but_unused"] = info.pop("gpu_model", None)
        info["gpu_model"], info["gpu_mem_total_gb"], info["gpu_driver"] = None, None, None
    return info

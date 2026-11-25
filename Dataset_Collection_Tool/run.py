#!/usr/bin/env python3
"""
Computation-latency data collection tool (launcher).

Run this file with any Python 3.9 or newer (Windows: run.bat, Linux/macOS: run.sh or python3 run.py);
the environment itself is built with a Python 3.9 - 3.13 found on the machine (3.12 preferred).
It will:
  1. detect the machine (CPU, NVIDIA GPU + driver, other GPUs, OS, Python versions), then ask for a
     machine id and whether to collect on CPU or GPU,
  2. create a private, isolated environment for that mode with the best available Python and
     install the PyTorch build that suits THIS machine (newest CUDA build the driver supports that
     actually runs on this GPU - e.g. RTX 50-series need recent builds, older GPUs older ones;
     the CPU build on CPU-only machines), plus the pinned packages,
  3. download the speech model once,
  4. start the collector (collector/collect.py) inside that environment; it waits until the machine
     is idle and on mains power, blocks sleep, and checks the data when it has finished,
  5. when the collection has finished, automatically delete the environment and downloaded model
     again, so the machine is back to its previous state, and pack the data into
     output/<machine_id>_<mode>.zip (the data in output/ is kept).
     If the collection is interrupted, the environment is kept so the same command can resume;
     `python3 run.py --cleanup` removes it at any time.
Questions take their default after 10 minutes without an answer, so an unattended start never hangs.
Everything shown on the screen is also saved to output/<machine_id>_<mode>/logs/.

Only the Python standard library is used here, so nothing needs to be installed beforehand.

Examples
  python3 run.py                                  # interactive
  python3 run.py --mode gpu --machine-id lab_gpu1  # non-interactive
  python3 run.py --mode cpu --machine-id lab_cpu1 --smoke   # 2-minute sanity check (20 files per request)
  python3 run.py --mode cpu --machine-id lab_cpu1 --dry-run # show the plan only
Any option not listed under "launcher options" is passed on to the collector
(see `python3 run.py --help-collector`).
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True   # leave no __pycache__ behind in the shared folder
sys.path.insert(0, str(ROOT))
from collector.config import FORMAT_VERSION  # noqa: E402  (standard library only)
from collector.prompt import ask as timed_ask  # noqa: E402

# PyTorch is NOT pinned: each machine gets the newest build that works on its hardware (the build
# used is recorded in machine_info.json and on every row). The other packages are pinned.
TORCH_INDEX = "https://download.pytorch.org/whl/"
# (wheel tag, minimum CUDA version the NVIDIA driver must support) - newest first
CUDA_WHEELS = [("cu130", (13, 0)), ("cu129", (12, 9)), ("cu128", (12, 8)), ("cu126", (12, 6)),
               ("cu124", (12, 4)), ("cu121", (12, 1)), ("cu118", (11, 8))]
PREFERRED_PY = [(3, 12), (3, 11), (3, 10), (3, 13), (3, 9)]   # PyTorch publishes wheels for these

SPEECH_MODEL_DIR = ROOT / "models" / "wav2vec2-large-960h"


class LauncherLog:
    """Copy of everything shown on the screen (including pip and collector output). Lines from before
    the output folder is known are kept and written first."""

    def __init__(self):
        self.pending, self.f, self.path = [], None, None

    def open(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", errors="replace")
        self.path = path
        self.f.write("".join(self.pending))
        self.f.flush()
        self.pending = []

    def write(self, text):
        if self.f:
            self.f.write(text)
            self.f.flush()
        else:
            self.pending.append(text)


LOG = LauncherLog()
QUESTION_TIMEOUT = 600.0   # s; then the default answer is taken


def say(msg=""):
    print(msg, flush=True)
    LOG.write(msg + "\n")


def die(msg):
    say(f"\n[ERROR] {msg}")
    sys.exit(1)


def ask(prompt, default=None, choices=None):
    ans = timed_ask(prompt, default, choices, timeout=QUESTION_TIMEOUT, say=say)
    LOG.write(f"  > {ans}\n")
    return ans


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


CACHE_DIR = None  # set in main(): <venv>/cache, so nothing is written outside this folder


def clean_env(**extra):
    """Environment for venv commands.
    * PYTHONPATH/PYTHONHOME are dropped so packages installed elsewhere on the machine
      (ROS, conda, ...) cannot leak into the isolated environment.
    * Every cache / config directory of the libraries we use is redirected into the environment
      folder and pip keeps no download cache, so deleting the folder restores the machine."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_TELEMETRY="1")
    if CACHE_DIR is not None:
        c = str(CACHE_DIR)
        env.update(
            XDG_CACHE_HOME=c, XDG_CONFIG_HOME=os.path.join(c, "config"),
            HF_HOME=os.path.join(c, "huggingface"), TORCH_HOME=os.path.join(c, "torch"),
            MPLCONFIGDIR=os.path.join(c, "matplotlib"), YOLO_CONFIG_DIR=os.path.join(c, "ultralytics"),
            CUDA_CACHE_PATH=os.path.join(c, "nv"),
        )
    env.update(extra)
    return env


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def cleanup(assume_yes=False):
    """Remove everything the generator installed or downloaded. The collected data in output/
    is kept. All caches were redirected into the environment folders, so nothing is left behind."""
    items = [
        (ROOT / ".venv-cpu", "Python environment for CPU mode (PyTorch etc.) + its caches"),
        (ROOT / ".venv-gpu", "Python environment for GPU mode (PyTorch etc.) + its caches"),
        (SPEECH_MODEL_DIR, "speech model downloaded on the first run"),
    ]
    venvs = (ROOT / ".venv-cpu", ROOT / ".venv-gpu")
    items += [(p, "temporary Python files") for p in ROOT.rglob("__pycache__")
              if p.is_dir() and not any(v in p.parents for v in venvs)]  # those inside a venv go with it
    items = [(p, what) for p, what in items if p.exists()]
    if not items:
        say("Nothing to clean: no environment or downloaded model found.")
        return
    sizes = [dir_size(p) for p, _ in items]
    say("\nCleanup - these were installed / downloaded by this tool and can be removed:")
    for (p, what), size in zip(items, sizes):
        if what != "temporary Python files":
            say(f"   {p.relative_to(ROOT)}/  {size / 2**30:5.2f} GB   {what}")
    n_tmp = sum(1 for _, what in items if what == "temporary Python files")
    if n_tmp:
        say(f"   {n_tmp} __pycache__/ folders (temporary Python files)")
    say(f"   total {sum(sizes) / 2**30:.1f} GB")
    say("KEPT: output/ (your collected data), the scripts, inputs/ and the YOLO weights in models/.")
    targets = [p for p, _ in items]
    if not assume_yes:
        if not sys.stdin.isatty():
            say("Run `python3 run.py --cleanup` to remove them.")
            return
        if input("Delete the items listed above? [Y/n]: ").strip().lower().startswith("n"):
            say("Kept. You can clean up later with:  python3 run.py --cleanup")
            return
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    say("Cleanup done - the machine is back to its previous state (only output/ was added).")


def stream(cmd, env=None, cwd=None):
    """Run a command, show its output as it comes and copy it into the launcher log.
    Returns its exit code. Ctrl+C reaches the command too; its last output is still shown."""
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env or clean_env(), cwd=cwd)
    while True:
        try:
            chunk = p.stdout.read1(4096)
        except KeyboardInterrupt:
            continue
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        LOG.write(text)
    while True:
        try:
            return p.wait()
        except KeyboardInterrupt:
            continue


def run(cmd, **kw):
    say("  $ " + " ".join(str(c) for c in cmd))
    rc = stream(cmd, env=kw.get("env"), cwd=kw.get("cwd"))
    if rc != 0:
        die(f"command failed (exit {rc}): {' '.join(str(c) for c in cmd)}")


def keep_awake_prefix():
    """Command prefix / setting that stops the machine from sleeping while collecting.
    Linux: systemd-inhibit (also blocks suspend on closing the lid; no root needed),
    macOS: caffeinate, Windows: SetThreadExecutionState for this process and its children."""
    if sys.platform.startswith("linux") and shutil.which("systemd-inhibit"):
        for what, note in (("sleep:idle:handle-lid-switch", "sleep, idle-suspend and lid-close suspend are blocked"),
                           ("sleep:idle", "sleep and idle-suspend are blocked (closing the lid may still suspend)")):
            cmd = ["systemd-inhibit", f"--what={what}", "--who=Edge latency data collector",
                   "--why=Measuring latency - sleeping would corrupt the data", "--mode=block"]
            if subprocess.run(cmd + ["true"], capture_output=True).returncode == 0:
                return cmd, note + " while collecting"
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        return ["caffeinate", "-dims"], "sleep is blocked while collecting (caffeinate)"
    if os.name == "nt":
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
            return [], "sleep is blocked while collecting"
        except Exception:  # noqa: BLE001
            pass
    return [], ("could NOT block sleep automatically - please disable sleep / suspend in the power settings "
                "(the script detects and redoes runs interrupted by sleep)")


def capture(cmd, timeout=60, env=None):
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


# ---------------------------------------------------------------------------------------------
# hardware detection
# ---------------------------------------------------------------------------------------------
def cpu_name():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        rc, out = capture(["sysctl", "-n", "machdep.cpu.brand_string"], 10)
        if rc == 0 and out.strip():
            return out.strip()
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        rc, out = capture(["lscpu"], 10)              # ARM boards have no "model name" in cpuinfo
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown CPU"


def nvidia_gpus():
    """([{name, mem_gb, driver, compute_cap}], (major, minor) CUDA version the driver supports)."""
    if not shutil.which("nvidia-smi"):
        return [], None
    rc, out = capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                       "--format=csv,noheader,nounits"], 30)
    if rc != 0:   # very old drivers do not know compute_cap
        rc, out = capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader,nounits"], 30)
    gpus = []
    for line in out.strip().splitlines() if rc == 0 else []:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            try:
                mem = float(p[1]) / 1024
            except ValueError:
                mem = 0.0
            gpus.append({"name": p[0], "mem_gb": mem, "driver": p[2], "compute_cap": p[3] if len(p) > 3 else "?"})
    cuda = None
    if gpus:
        rc, out = capture(["nvidia-smi"], 30)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            cuda = (int(m.group(1)), int(m.group(2)))
    return gpus, cuda


def other_gpus():
    """Non-NVIDIA GPUs (reported only: GPU mode needs an NVIDIA GPU)."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return ["Apple Silicon GPU"]
    names = []
    if sys.platform.startswith("linux") and shutil.which("lspci"):
        rc, out = capture(["lspci"], 10)
        names = [l.split(": ", 1)[-1] for l in out.splitlines()
                 if re.search(r"VGA|3D controller|Display", l) and "NVIDIA" not in l.upper()
                 and re.search(r"AMD|ATI|Radeon|Intel", l, re.I)]
    return names


def on_battery():
    try:
        if sys.platform.startswith("linux"):
            for ac in Path("/sys/class/power_supply").glob("*"):
                if (ac / "type").exists() and (ac / "type").read_text().strip() == "Mains":
                    return (ac / "online").read_text().strip() == "0"
        elif os.name == "nt":
            import ctypes

            class SPS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                            ("BatteryLifePercent", ctypes.c_byte), ("Reserved1", ctypes.c_byte),
                            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            st = SPS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
                return st.ACLineStatus == 0
        elif sys.platform == "darwin":
            rc, out = capture(["pmset", "-g", "batt"], 10)
            return "Battery Power" in out
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------------------------
def python_version(py):
    rc, out = capture([py, "-c", "import sys; print('@@', sys.version_info[0], sys.version_info[1])"], 30)
    m = re.search(r"@@ (\d+) (\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_base_python():
    """(path, version) of the best Python on this machine to build the environment with
    (one that PyTorch publishes wheels for; 3.12 preferred)."""
    cands = []
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    cands.append(getattr(sys, "_base_executable", None) if in_venv else sys.executable)
    if os.name == "nt" and shutil.which("py"):
        for v in PREFERRED_PY:
            rc, out = capture(["py", f"-{v[0]}.{v[1]}", "-c", "import sys; print('@@', sys.executable)"], 30)
            m = re.search(r"@@ (.+)", out)
            if rc == 0 and m:
                cands.append(m.group(1).strip())
    for v in PREFERRED_PY:
        exe = shutil.which(f"python{v[0]}.{v[1]}")
        if exe:
            cands.append(exe)
    for name in ("python3", "python"):
        exe = shutil.which(name)
        if exe:
            cands.append(exe)
    seen, scored = set(), []
    for c in filter(None, cands):
        key = str(c).lower()
        if key in seen or "windowsapps" in key:      # skip the Microsoft Store stub
            continue
        seen.add(key)
        v = python_version(c)
        if v and (3, 9) <= v < (3, 14):
            scored.append((PREFERRED_PY.index(v) if v in PREFERRED_PY else 99, c, v))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored[0][2]


def pip(py, args):
    say("  $ pip " + " ".join(str(a) for a in args))
    return stream([py, "-m", "pip", "--disable-pip-version-check", *args])


GPU_TEST = r"""
import json, torch
r = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "available": torch.cuda.is_available()}
if r["available"]:
    try:
        r["device"] = torch.cuda.get_device_name(0)
        x = torch.randn(256, 256, device="cuda"); float((x @ x).sum())                 # matrix multiply
        c = torch.nn.Conv2d(3, 8, 3).cuda(); float(c(torch.randn(1, 3, 64, 64, device="cuda")).sum())  # cuDNN conv
        torch.cuda.synchronize(); r["test"] = "ok"
    except Exception as e:
        r["test"] = (type(e).__name__ + ": " + str(e))[:300]
print("@@" + json.dumps(r))
"""


def torch_report(py):
    rc, out = capture([py, "-c", GPU_TEST], 600, env=clean_env())
    for line in out.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    return {"error": out.strip()[-300:]}


def install_torch(py, mode, driver_cuda):
    """GPU: try the CUDA builds newest-first (only those the driver supports); keep the first one
    whose real GPU test passes - new GPUs (e.g. RTX 50-series) need recent builds, old GPUs may
    only work with older ones. CPU: the CPU build."""
    if mode == "cpu":
        extra = [] if sys.platform == "darwin" else ["--index-url", TORCH_INDEX + "cpu"]
        say("\n== Installing the CPU build of PyTorch ==")
        if pip(py, ["install", "--progress-bar", "off", "--upgrade", "torch", "torchvision", *extra]) != 0:
            die("could not install PyTorch (see the messages above).")
        return "cpu"
    tags = [tag for tag, need in CUDA_WHEELS if driver_cuda is None or need <= driver_cuda]
    if not tags:
        die(f"the NVIDIA driver only supports CUDA {driver_cuda[0]}.{driver_cuda[1]}; PyTorch needs >= 11.8. "
            "Update the NVIDIA driver, or rerun and choose 'cpu'.")
    for tag in tags:
        say(f"\n== Installing PyTorch built for CUDA ({tag}) - about 3 GB ==")
        if pip(py, ["install", "--progress-bar", "off", "--upgrade", "torch", "torchvision",
                    "--index-url", TORCH_INDEX + tag]) != 0:
            say(f"  no {tag} build for this Python / OS - trying the next one")
            continue
        rep = torch_report(py)
        if rep.get("available") and rep.get("test") == "ok":
            say(f"  PyTorch {rep['torch']} (CUDA {rep['cuda_build']}) works on {rep.get('device')}")
            return tag
        say(f"  this build cannot use this GPU ({rep.get('test') or rep.get('error') or 'CUDA not available'})"
            " - trying an older build")
        pip(py, ["uninstall", "-y", "torch", "torchvision"])
    die("no PyTorch CUDA build works on this GPU. Update the NVIDIA driver, or rerun and choose 'cpu'.")


def create_venv(venv_dir):
    found = find_base_python()
    if not found:
        die("no usable Python 3.9 - 3.13 found. Install Python 3.12 "
            "(python.org, or 'sudo apt install python3.12 python3.12-venv') and run again.")
    base, ver = found
    say(f"  creating {venv_dir.name} with Python {ver[0]}.{ver[1]} ({base})")
    py = venv_python(venv_dir)
    r = subprocess.run([str(base), "-m", "venv", str(venv_dir)], env=clean_env())
    if r.returncode != 0 or not py.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
        if capture([base, "-m", "virtualenv", "--version"], 30)[0] == 0:
            say("  'venv' failed - trying 'virtualenv' instead")
            if subprocess.run([str(base), "-m", "virtualenv", str(venv_dir)], env=clean_env()).returncode == 0 \
                    and py.exists():
                return py
        hint = (f"\nOn Ubuntu/Debian install it with:  sudo apt install python{ver[0]}.{ver[1]}-venv"
                if sys.platform.startswith("linux") else "")
        die("could not create a virtual environment." + hint)
    return py


IMPORT_CHECK = ("import torch, torchvision, transformers, soundfile, cv2, psutil, pynvml, ultralytics, "
                "scipy, pandas; print('imports ok')")


def imports_ok(py):
    rc, out = capture([py, "-c", IMPORT_CHECK], 300, env=clean_env())
    return "imports ok" in out, out


def setup_env(mode, driver_cuda, gpu_name, force):
    venv_dir = ROOT / f".venv-{mode}"
    py = venv_python(venv_dir)
    req = (ROOT / "requirements.txt").read_text()
    stamp_val = hashlib.sha256(f"{req}|{mode}|{driver_cuda}|{gpu_name}".encode()).hexdigest()
    stamp = venv_dir / ".setup_complete"

    if force and venv_dir.exists():
        say(f"Removing existing environment {venv_dir.name} ...")
        shutil.rmtree(venv_dir)

    if stamp.exists() and stamp.read_text().strip() == stamp_val and py.exists():
        rep = torch_report(py)
        if "error" not in rep and (mode == "cpu" or rep.get("test") == "ok"):
            ok, out = imports_ok(py)
            if not ok:
                # PyTorch works but another package is broken / missing: reinstall the packages only
                say("Some packages of the environment are broken - reinstalling them:\n" + out[-400:])
                pip(py, ["install", "--progress-bar", "off", "-r", ROOT / "requirements.txt",
                         "-c", venv_dir / "constraints.txt"])
                ok, out = imports_ok(py)
            if ok:
                say(f"Environment {venv_dir.name} already set up (PyTorch {rep.get('torch')}) - skipping installation.")
                return py
        say("Existing environment does not work any more - rebuilding it.")
        shutil.rmtree(venv_dir, ignore_errors=True)

    say(f"\n== Creating the environment {venv_dir.name} ==")
    if not py.exists():
        py = create_venv(venv_dir)
    pip(py, ["install", "--progress-bar", "off", "--upgrade", "pip", "wheel"])
    tag = install_torch(py, mode, driver_cuda)

    # keep the PyTorch build that was chosen for this machine while installing the other packages
    rc, out = capture([py, "-c", "import torch, torchvision; print('@@', torch.__version__, torchvision.__version__)"],
                      120, env=clean_env())
    m = re.search(r"@@ (\S+) (\S+)", out)
    if not m:
        die("PyTorch was installed but cannot be imported:\n" + out[-500:])
    constraints = venv_dir / "constraints.txt"
    constraints.write_text(f"torch=={m.group(1)}\ntorchvision=={m.group(2)}\n")
    say("\n== Installing the other packages ==")
    if pip(py, ["install", "--progress-bar", "off", "-r", ROOT / "requirements.txt", "-c", constraints]) != 0:
        die("installing the packages failed (see the messages above).")

    say("\n== Checking the installation ==")
    ok, out = imports_ok(py)
    if not ok:
        die("package check failed:\n" + out[-800:])
    rep = torch_report(py)
    say(f"  PyTorch {rep.get('torch')} (build: {tag}, CUDA {rep.get('cuda_build') or '-'}) | "
        + (f"GPU test: {rep.get('test')} on {rep.get('device')}" if mode == "gpu" else "CPU mode"))
    if mode == "gpu" and rep.get("test") != "ok":
        die("GPU mode selected but PyTorch cannot use the GPU. Check the NVIDIA driver, or rerun with --mode cpu.")
    stamp.write_text(stamp_val)
    return py


def ensure_models(py: Path):
    for w in ("yolov5s.pt", "yolov5s-seg.pt"):
        if not (ROOT / "models" / w).exists():
            die(f"models/{w} is missing from the shared folder.")
    if (SPEECH_MODEL_DIR / ".download_complete").exists():
        return
    say("\n== Downloading the speech model facebook/wav2vec2-large-960h (~1.3 GB, once) ==")
    run([py, "-m", "collector.download_models", "--out", SPEECH_MODEL_DIR], cwd=ROOT)


def output_dir(machine_id, mode, passthrough):
    """The collector's output folder for these options (as collector/collect.py chooses it)."""
    for i, a in enumerate(passthrough):
        if a == "--out-dir" and i + 1 < len(passthrough):
            return Path(passthrough[i + 1])
        if a.startswith("--out-dir="):
            return Path(a.split("=", 1)[1])
    return ROOT / "output" / (f"{machine_id}_{mode}" + ("_smoke" if "--smoke" in passthrough else ""))


def move_old_output(out):
    """Data collected with an earlier version of this tool cannot be continued (different columns):
    keep it under another name, so the new collection starts in a clean folder."""
    info = out / "machine_info.json"
    if not info.exists():
        return
    try:
        old = json.loads(info.read_text()).get("format_version")
    except (OSError, ValueError):
        old = None
    if old == FORMAT_VERSION:
        return      # same version: the collection resumes in it
    dest = out.with_name(f"{out.name}_old_{time.strftime('%Y%m%d_%H%M%S')}")
    out.rename(dest)
    say(f"  note    : {out.name}/ held data from an earlier version of this tool; kept as {dest.name}/")


def check_disk(mode):
    venv_ready = (ROOT / f".venv-{mode}" / ".setup_complete").exists()
    model_ready = (SPEECH_MODEL_DIR / ".download_complete").exists()
    need = 1.0 + (0 if venv_ready else (8.0 if mode == "gpu" else 3.0)) + (0 if model_ready else 1.5)
    free = shutil.disk_usage(ROOT).free / 2**30
    say(f"  disk    : {free:.0f} GB free here, about {need:.0f} GB needed")
    if free < need:
        die(f"not enough free disk space: {free:.1f} GB free, about {need:.0f} GB needed "
            "(environment, speech model and data). Free some space and run again.")


def zip_output(out):
    """Pack the output folder into output/<name>.zip for sending back (.gz files are stored as they are)."""
    zip_path = out.with_name(out.name + ".zip")
    tmp = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f.relative_to(out.parent).as_posix(),
                        compress_type=zipfile.ZIP_STORED if f.suffix == ".gz" else zipfile.ZIP_DEFLATED)
    os.replace(tmp, zip_path)
    return zip_path


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(
        description="Computation-latency data collection tool launcher",
        epilog="Unrecognised options are forwarded to the collector; see --help-collector.")
    g = ap.add_argument_group("launcher options")
    g.add_argument("--mode", choices=["cpu", "gpu"], help="run the tasks on the CPU only, or on the GPU")
    g.add_argument("--machine-id", help="short unique name for this machine, e.g. lab_i7_cpu (used in file names)")
    g.add_argument("--reinstall", action="store_true", help="delete and recreate the virtual environment")
    g.add_argument("--setup-only", action="store_true", help="install everything, then stop")
    g.add_argument("--help-collector", action="store_true", help="show the collector's options and exit")
    g.add_argument("--cleanup", action="store_true",
                   help="delete the environment(s) and downloaded model (keeps output/), then exit")
    g.add_argument("--keep-env", action="store_true",
                   help="do not remove the environment after the collection (by default it is removed automatically)")
    g.add_argument("--yes", action="store_true", help="with --cleanup: do not ask for confirmation")
    args, passthrough = ap.parse_known_args()

    if args.cleanup:
        cleanup(assume_yes=args.yes)
        return

    if sys.version_info < (3, 9):
        die(f"Python {platform.python_version()} is too old to start this script; use Python 3.9 or newer "
            "(3.12 recommended).")

    say("=" * 70)
    say(" Computation-latency data collection tool")
    say("=" * 70)

    # ---- what is this machine? ----
    gpus, driver_cuda = nvidia_gpus()
    others = other_gpus()
    say(f"  machine : {socket.gethostname()}  ({platform.system()} {platform.release()}, {platform.machine()})")
    say(f"  CPU     : {cpu_name()}  ({os.cpu_count()} logical cores)")
    for g in gpus:
        say(f"  GPU     : {g['name']}  {g['mem_gb']:.0f} GB, compute capability {g['compute_cap']}, "
            f"driver {g['driver']}" + (f" (CUDA <= {driver_cuda[0]}.{driver_cuda[1]})" if driver_cuda else ""))
    for o in others:
        say(f"  GPU     : {o}  (not NVIDIA - only CPU mode is possible with it)")
    if not gpus and not others:
        say("  GPU     : none found")
    found_py = find_base_python()
    say(f"  Python  : launcher {platform.python_version()}; environment will use "
        + (f"{found_py[1][0]}.{found_py[1][1]} ({found_py[0]})" if found_py else "NONE FOUND (3.9 - 3.13 needed)"))
    if on_battery():
        say("  WARNING : running on battery - please plug the machine in (battery saving changes the timings);"
            " runs wait until it is plugged in")

    mode = args.mode or ask("Run the tasks on CPU only or with GPU acceleration? (cpu/gpu)",
                            default="gpu" if gpus else "cpu", choices=["cpu", "gpu"])
    if mode == "gpu" and not gpus:
        die("GPU mode needs an NVIDIA GPU with its driver installed (nvidia-smi must work). "
            "Rerun and choose 'cpu'.")
    machine_id = args.machine_id
    if not machine_id:
        default = re.sub(r"[^A-Za-z0-9_-]+", "_", socket.gethostname())[:30] or "machine"
        machine_id = ask("Short unique name for this machine (letters, digits, _ or -)", default=default)
    machine_id = re.sub(r"[^A-Za-z0-9_-]+", "_", machine_id)

    # from here on, everything shown is also saved in the output folder (sent back with the data)
    out = output_dir(machine_id, mode, passthrough)
    if not (args.help_collector or "--dry-run" in passthrough):
        move_old_output(out)
        LOG.open(out / "logs" / f"launcher_{time.strftime('%Y%m%d_%H%M%S')}.log")
    check_disk(mode)

    global CACHE_DIR
    CACHE_DIR = ROOT / f".venv-{mode}" / "cache"

    py = setup_env(mode, driver_cuda, gpus[0]["name"] if (mode == "gpu" and gpus) else "-", args.reinstall)
    ensure_models(py)
    if args.setup_only:
        say("\nSetup finished. Start collection later with:  python3 run.py "
            f"--mode {mode} --machine-id {machine_id}")
        return

    cmd = [py, "-m", "collector.collect", "--mode", mode, "--machine-id", machine_id] + passthrough
    if args.help_collector:
        cmd = [py, "-m", "collector.collect", "--help"]
    if not args.help_collector:
        prefix, note = keep_awake_prefix()
        cmd = prefix + cmd
        say(f"\nKeep-awake: {note}.")
        say("Do not shut the machine down; if it happens, rerun the same command to resume.")
    say("\n== Starting collection ==")
    rc = stream(cmd, env=clean_env(), cwd=ROOT)
    # collector exit codes: 0 = complete (every run ok, or did not fit in memory), 3 = finished but some
    # runs still failed after the retry passes; anything else = interrupted or crashed
    if rc not in (0, 3):
        say("\nCollection stopped before the end. The environment is kept so you can resume:\n"
            "  rerun the same command to continue where it left off, or\n"
            "  run  python3 run.py --cleanup  to remove the installed files instead.")
        sys.exit(rc)
    check_only = "--smoke" in passthrough or "--dry-run" in passthrough or args.help_collector
    if check_only:
        say("\nCheck finished - the environment is kept for the full collection "
            "(run.py removes it automatically after the full collection; or run  python3 run.py --cleanup).")
        sys.exit(rc)
    if args.keep_env:
        say("\n--keep-env given: the environment is kept. Remove it later with  python3 run.py --cleanup")
    else:
        if rc == 3:
            say("\nNote: some runs failed (see runs.csv). Retrying them later needs a new setup "
                "(rerun the same command; completed runs are skipped).")
        say("\n== Cleaning up (automatic) ==")
        cleanup(assume_yes=True)
    say("\n== Packing the data ==")
    zip_path = zip_output(out)
    say(f"  {zip_path}  ({zip_path.stat().st_size / 2**20:.0f} MB)")
    say(f"\nDone. Please send back {zip_path.name} (it contains the folder {out.name}/)."
        + ("" if rc == 0 else " Some runs are missing - see the end of collect.log."))
    sys.exit(rc)


if __name__ == "__main__":
    main()

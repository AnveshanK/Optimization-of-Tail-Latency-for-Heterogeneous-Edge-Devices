"""Static configuration shared by the collector and its workers (no heavy imports here)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output"

# Task letters follow the earlier project codebase: s = speech, d = detect, p = predict (segmentation).
TASKS = {
    "s": {"name": "speech", "input_dir": INPUTS / "audio", "ext": (".flac", ".wav")},
    "d": {"name": "detect", "input_dir": INPUTS / "images", "ext": (".jpg", ".jpeg", ".png")},
    "p": {"name": "predict", "input_dir": INPUTS / "images", "ext": (".jpg", ".jpeg", ".png")},
}
TASK_ORDER = "sdp"  # canonical letter order inside a combination string
FORMAT_VERSION = "2026-10-06"   # layout of the output files; one dataset folder never mixes two versions


def list_inputs(task):
    cfg = TASKS[task]
    return sorted(str(p) for p in cfg["input_dir"].iterdir() if p.suffix.lower() in cfg["ext"])


def canonical(combo):
    """'psp' -> 'spp'. Order is irrelevant when tasks run concurrently."""
    return "".join(sorted(combo, key=TASK_ORDER.index))

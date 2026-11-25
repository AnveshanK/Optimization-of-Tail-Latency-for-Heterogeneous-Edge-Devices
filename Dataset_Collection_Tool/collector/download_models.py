"""Download the Wav2Vec2 speech model once into models/ so collection runs offline."""

import argparse
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

REPO = "facebook/wav2vec2-large-960h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    files = list_repo_files(REPO)
    weights = ["model.safetensors"] if "model.safetensors" in files else ["pytorch_model.bin"]
    snapshot_download(REPO, local_dir=str(out), allow_patterns=["*.json", "*.txt"] + weights)
    (out / ".download_complete").write_text("ok\n")
    print(f"Saved {REPO} to {out}")


if __name__ == "__main__":
    main()

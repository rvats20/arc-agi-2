#!/usr/bin/env python3
"""Helpers to bundle Qwen2.5-VL-7B-Instruct (4-bit) as a Kaggle dataset.

This script does NOT download 15GB. It prepares the dataset folder + metadata and
pushes it once YOU have placed the 4-bit weights there. Steps:

1. During DEV (on a machine with internet), get the 4-bit weights, e.g.:
     pip install huggingface_hub
     huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-4bit \\
         --local-dir qwen25vl_4bit
   (any 4-bit build works; transformers loads it via AutoProcessor/from_pretrained.)

2. Put the weights in ./qwen25vl_4bit and run this script:
     python make_kaggle_model_dataset.py --push
   It writes dataset-metadata.json and runs `kaggle datasets create`.

The notebook expects the dataset mounted at /kaggle/input/models/Qwen2.5-VL-7B-Instruct-4bit
(change QWEN_PATH in the notebook config cell if you name it differently).
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "qwen25vl_4bit"
SLUG = "qwen25vl-7b-instruct-4bit"
USER = "rahul"  # <-- set your Kaggle username here
TITLE = "Qwen2.5-VL-7B-Instruct 4bit"


def ensure_dir():
    DATASET_DIR.mkdir(exist_ok=True)
    print(f"dataset dir: {DATASET_DIR}")


def write_metadata():
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "other"}],
        "description": "Qwen2.5-VL-7B-Instruct in 4-bit (transformers loadable) for ARC-AGI-2.",
    }
    (DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote dataset-metadata.json")


def push(kaggle_bin: str):
    # find a kaggle CLI binary
    import shutil as _s
    kb = kaggle_bin or (_s.which("kaggle") or "")
    if not kb:
        # fall back to a known venv binary path pattern
        kb = str(HERE / ".venv/bin/kaggle")
    print(f"using kaggle CLI: {kb}")
    r = subprocess.run([kb, "datasets", "create", "-p", str(DATASET_DIR)], check=False)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="create the dataset on Kaggle")
    ap.add_argument("--kaggle-bin", default="", help="path to kaggle CLI if not on PATH")
    args = ap.parse_args()
    if not (DATASET_DIR / "dataset-metadata.json").exists():
        ensure_dir(); write_metadata()
    weights = list(DATASET_DIR.glob("*.safetensors")) + list(DATASET_DIR.glob("*.bin"))
    if not weights:
        print("WARNING: no model weights found in", DATASET_DIR)
        print("Download them first, e.g.:")
        print("  huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-4bit --local-dir", DATASET_DIR)
        sys.exit(2)
    print(f"found {len(weights)} weight file(s); total size:")
    total = sum(f.stat().st_size for f in weights)
    print(f"  {total/1e9:.1f} GB")
    if args.push:
        rc = push(args.kaggle_bin)
        print("kaggle datasets create rc:", rc)


if __name__ == "__main__":
    main()

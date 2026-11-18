"""Validated entry point for the BISWA AI Wan2.2 LoRA trainer.

The actual training is performed by training/train_wan22_ti2v_lora.sh on a
machine with a compatible CUDA GPU and an authorized training dataset.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "training" / "config.yaml"
SCRIPT_PATH = ROOT / "training" / "train_wan22_ti2v_lora.sh"


def main() -> None:
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Training config not found: {CONFIG_PATH}")
    if not SCRIPT_PATH.is_file():
        raise SystemExit(f"Training launcher not found: {SCRIPT_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    required = (
        "base_model", "trainer", "dataset_dir", "metadata_file",
        "output_dir", "height", "width", "frames", "epochs",
        "learning_rate", "lora_rank",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit("Missing training config values: " + ", ".join(missing))

    dataset_dir = (ROOT / config["dataset_dir"]).resolve()
    metadata_file = (ROOT / config["metadata_file"]).resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    if not metadata_file.is_file():
        raise SystemExit(
            f"Dataset metadata not found: {metadata_file}. "
            "Run training/prepare_dataset.py first."
        )

    env = os.environ.copy()
    env.update({
        "BISWA_DATA_DIR": str(dataset_dir),
        "BISWA_OUTPUT_DIR": str((ROOT / config["output_dir"]).resolve()),
        "BISWA_HEIGHT": str(config["height"]),
        "BISWA_WIDTH": str(config["width"]),
        "BISWA_FRAMES": str(config["frames"]),
        "BISWA_DATASET_REPEAT": str(config.get("dataset_repeat", 1)),
        "BISWA_EPOCHS": str(config["epochs"]),
        "BISWA_LR": str(config["learning_rate"]),
        "BISWA_LORA_RANK": str(config["lora_rank"]),
    })

    print(f"Starting {config['trainer']} training for {config['base_model']}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {env['BISWA_OUTPUT_DIR']}")
    subprocess.run(["bash", str(SCRIPT_PATH)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

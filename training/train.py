"""Minimal training entry point.

This is intentionally a framework, not a claim that a large video model can
be trained on GitHub Actions or a phone. Real training needs suitable GPU
hardware and a properly licensed dataset.
"""
from pathlib import Path
import yaml


def main():
    cfg = yaml.safe_load(Path("training/config.yaml").read_text())
    if not cfg.get("model_name"):
        raise SystemExit("Set training/config.yaml:model_name first.")
    print("Training configuration loaded:", cfg)
    print("Connect the checkpoint-specific trainer here.")


if __name__ == "__main__":
    main()

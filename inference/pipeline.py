from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

WAN_TASK = os.getenv("BISWA_WAN_TASK", "ti2v-5B")
WAN_SIZE = os.getenv("BISWA_WAN_SIZE", "1280*704")


def generate_video(
    image_path: Path,
    prompt: str,
    workdir: Path,
    model_dir: Optional[Path] = None,
) -> Path:
    """Run the official Wan2.2 TI2V-5B image-to-video inference script.

    BISWA AI currently adapts an open-source base checkpoint; it does not yet
    contain independently trained BISWA model weights.
    """
    image_path = Path(image_path)
    workdir = Path(workdir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not prompt or not prompt.strip():
        raise ValueError("A non-empty motion prompt is required.")

    model_dir = Path(
        model_dir or os.getenv("BISWA_WAN_MODEL_DIR", "./models/Wan2.2-TI2V-5B")
    )
    official_dir = Path(os.getenv("BISWA_WAN_SOURCE_DIR", "./third_party/Wan2.2"))
    generate_script = official_dir / "generate.py"

    if not generate_script.is_file():
        raise FileNotFoundError(
            f"Wan2.2 generate.py not found at {generate_script}. "
            "Clone Wan-Video/Wan2.2 into the configured source directory."
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Wan2.2 checkpoint directory not found: {model_dir}. "
            "Download Wan-AI/Wan2.2-TI2V-5B before starting the API."
        )

    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / "biswa_ai_result.mp4"
    command = [
        sys.executable,
        str(generate_script),
        "--task", WAN_TASK,
        "--size", WAN_SIZE,
        "--ckpt_dir", str(model_dir),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--image", str(image_path),
        "--prompt", prompt.strip(),
        "--save_file", str(output),
    ]

    timeout = int(os.getenv("BISWA_GENERATION_TIMEOUT_SECONDS", "7200"))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "No process output")[-4000:]
        raise RuntimeError(f"Wan2.2 exited with code {completed.returncode}: {details}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Wan2.2 finished without creating a non-empty MP4 file.")
    return output

from pathlib import Path
import os
import subprocess
import sys
from typing import Optional

WAN_REPO = os.getenv("BISWA_WAN_REPO", "Wan-Video/Wan2.2")
WAN_TASK = os.getenv("BISWA_WAN_TASK", "ti2v-5B")
WAN_SIZE = os.getenv("BISWA_WAN_SIZE", "1280*704")


def generate_video(
    image_path: Path,
    prompt: str,
    workdir: Path,
    model_dir: Optional[Path] = None,
) -> Path:
    """Generate an image-to-video prototype with Wan2.2 TI2V-5B.

    This is an adapter around the open-source base model, not BISWA AI's
    trained weights yet. The official Wan2.2 code and checkpoint must be
    available on the GPU machine running this function.
    """
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    model_dir = model_dir or Path(
        os.getenv("BISWA_WAN_MODEL_DIR", "./models/Wan2.2-TI2V-5B")
    )
    official_dir = Path(
        os.getenv("BISWA_WAN_SOURCE_DIR", "./third_party/Wan2.2")
    )
    output = workdir / "biswa_ai_result.mp4"

    command = [
        sys.executable,
        str(official_dir / "generate.py"),
        "--task",
        WAN_TASK,
        "--size",
        WAN_SIZE,
        "--ckpt_dir",
        str(model_dir),
        "--offload_model",
        "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--save_file",
        str(output),
    ]

    subprocess.run(command, check=True)
    if not output.exists():
        raise RuntimeError("Wan2.2 finished without creating an output video.")
    return output

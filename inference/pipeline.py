from pathlib import Path


def generate_video(image_path: Path, prompt: str, workdir: Path) -> Path:
    """Adapter boundary for the selected open-source video model.

    Replace this function with the model-specific inference pipeline.
    User input/output remains inside a temporary directory and is removed
    when the API request finishes.
    """
    raise NotImplementedError(
        "Connect a licensed/open-source image-to-video checkpoint here."
    )

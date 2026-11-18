from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTask
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from inference.pipeline import generate_video

app = FastAPI(title="BISWA AI API", version="0.2.0")

MAX_IMAGE_BYTES = int(os.getenv("BISWA_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@app.get("/health")
def health():
    return {"ok": True, "service": "biswa-ai", "version": app.version}


@app.post("/v1/image-to-video")
async def image_to_video(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=3, max_length=1500),
):
    """Generate a video from one uploaded image and a motion prompt.

    The generated file is removed only after FastAPI has finished streaming it.
    """
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    workdir = Path(tempfile.mkdtemp(prefix="biswa-ai-"))
    src = workdir / f"input{suffix}"
    output = workdir / "biswa_ai_result.mp4"

    try:
        total = 0
        with src.open("wb") as target:
            while chunk := await image.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
                    )
                target.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")

        try:
            with Image.open(src) as uploaded_image:
                if uploaded_image.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=415, detail="Unsupported image format.")
                uploaded_image.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

        try:
            result = generate_video(src, prompt.strip(), workdir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="The Wan2.2 source code or model checkpoint is not installed on this server.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Video generation failed: {type(exc).__name__}. Check the server logs.",
            ) from exc

        if not result.is_file():
            raise HTTPException(status_code=500, detail="The video generator did not create an output file.")

        return FileResponse(
            path=result,
            media_type="video/mp4",
            filename="biswa-ai-result.mp4",
            background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
        )
    except Exception:
        # Keep the work directory only when a response is streaming successfully.
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        raise
    finally:
        await image.close()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import shutil

from inference.pipeline import generate_video

app = FastAPI(title="BISWA AI API", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True, "service": "biswa-ai"}

@app.post("/v1/image-to-video")
async def image_to_video(image: UploadFile = File(...), prompt: str = Form("natural dance motion")):
    suffix = Path(image.filename or "input.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        with src.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        output = generate_video(src, prompt, Path(tmp))
        return FileResponse(output, media_type="video/mp4", filename="result.mp4")

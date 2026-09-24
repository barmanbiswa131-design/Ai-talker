"""Create a simple video metadata CSV for BISWA AI training.

This does not upload or copy the media. It only scans the local dataset
directory and writes relative paths plus starter captions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def caption_for(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ")
    return f"a realistic video of {name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.input_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit("No video files found.")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "text"])
        for video in videos:
            writer.writerow([
                video.relative_to(root.parent).as_posix(),
                caption_for(video),
            ])

    print(f"Wrote {len(videos)} videos to {output}")


if __name__ == "__main__":
    main()

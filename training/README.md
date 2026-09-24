# BISWA AI training

This directory contains the first real BISWA AI fine-tuning pipeline.

## What this trains

The first target is a **Wan2.2 TI2V-5B LoRA**. The base model remains the Wan2.2 checkpoint; the LoRA learns BISWA-specific behavior from a properly licensed dataset.

Wan2.2 TI2V-5B supports text+image-to-video and 720P/24fps inference. The DiffSynth-Studio training framework documents a Wan2.2 TI2V-5B LoRA recipe and supports LoRA parameters such as rank, target modules, gradient checkpointing and multi-GPU training. citeturn1search4turn1search1

## Dataset layout

Create:

```
data/biswa_video/
  videos/
    clip_001.mp4
    clip_002.mp4
  metadata.csv
```

`metadata.csv`:

```csv
video,text
videos/clip_001.mp4,a person performs a smooth dance in a natural outdoor setting
videos/clip_002.mp4,a person walks naturally toward the camera
```

Only use media you own or have permission/licensing to train on.

## Prepare metadata

```bash
python training/prepare_dataset.py --input_dir data/biswa_video/videos --output data/biswa_video/metadata.csv
```

The generated captions are intentionally generic. For best results, manually improve the captions so they describe the actual motion, camera, subject and scene.

## Start LoRA training

```bash
bash training/train_wan22_ti2v_lora.sh
```

The script clones DiffSynth-Studio, installs its training dependencies, and launches the Wan2.2 TI2V-5B LoRA recipe.

The default settings are conservative: 480x832, 49 frames, rank 32, 5 epochs. Higher resolution/longer clips require substantially more compute.

## Full training later

LoRA is the first stage because it is much cheaper than changing the whole 5B model. The same DiffSynth training framework also exposes full model training through `--trainable_models`; a future full-training profile can be added after the LoRA checkpoint has been evaluated. citeturn1search5

## Important

The repository does not claim that training has already run. GitHub stores the training code/configuration; the actual training must run on a compatible GPU machine.

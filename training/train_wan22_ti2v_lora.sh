#!/usr/bin/env bash
set -euo pipefail

# BISWA AI — Wan2.2 TI2V-5B LoRA training
# Actual training requires a CUDA GPU. This script does not pretend to train
# inside GitHub; it prepares and launches the real DiffSynth-Studio trainer.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-$ROOT_DIR/third_party/DiffSynth-Studio}"
DATA_DIR="${BISWA_DATA_DIR:-$ROOT_DIR/data/biswa_video}"
OUTPUT_DIR="${BISWA_OUTPUT_DIR:-$ROOT_DIR/models/BISWA-Wan2.2-TI2V-5B-LoRA}"

HEIGHT="${BISWA_HEIGHT:-480}"
WIDTH="${BISWA_WIDTH:-832}"
FRAMES="${BISWA_FRAMES:-49}"
REPEAT="${BISWA_DATASET_REPEAT:-100}"
EPOCHS="${BISWA_EPOCHS:-5}"
LR="${BISWA_LR:-1e-4}"
RANK="${BISWA_LORA_RANK:-32}"

if [ ! -f "$DATA_DIR/metadata.csv" ]; then
  echo "Missing $DATA_DIR/metadata.csv"
  echo "Run:"
  echo "  python training/prepare_dataset.py --input_dir $DATA_DIR/videos --output $DATA_DIR/metadata.csv"
  exit 1
fi

if [ ! -d "$DIFFSYNTH_DIR" ]; then
  git clone --depth 1 https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
fi

python -m pip install -U accelerate modelscope
python -m pip install -e "$DIFFSYNTH_DIR"

cd "$DIFFSYNTH_DIR"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATA_DIR" \
  --dataset_metadata_path "$DATA_DIR/metadata.csv" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$FRAMES" \
  --dataset_repeat "$REPEAT" \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate "$LR" \
  --num_epochs "$EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "$RANK" \
  --extra_inputs "input_image"

echo
echo "BISWA AI LoRA training finished."
echo "Checkpoint directory: $OUTPUT_DIR"

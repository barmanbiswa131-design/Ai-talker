# BISWA AI video prototype

The first practical prototype uses **Wan2.2 TI2V-5B** as the open-source base model for text+image-to-video. The goal is to generate one test clip before attempting any BISWA-specific training.

Wan2.2 TI2V-5B supports text-to-video and image-to-video at 720p/24fps and is documented by its authors as requiring about 24 GB VRAM for the official single-GPU setup. It is therefore not expected to run on a phone. A free GPU notebook may work when suitable GPU time is available.

## Important

- This is a prototype using an existing open-source base model.
- It is **not** the final BISWA-trained model.
- Do not put private user media into a public repository.
- Use only images/videos you have the right to process.
- Model license terms must be checked before commercial deployment.

## Test flow

1. Open the Colab notebook in `notebooks/BISWA_AI_Wan22_Test.ipynb`.
2. Enable a GPU runtime.
3. Upload one image.
4. Enter a motion prompt.
5. Run the cells.
6. Download the generated short MP4.
7. Evaluate face, hands, body motion, background consistency and prompt following.

If the prototype quality is useful, the next stage is BISWA-specific fine-tuning/LoRA and evaluation rather than training a giant model from zero.

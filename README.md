# BISWA AI

A model-first project for building a future BISWA AI video-generation model.

## Current milestone: real prototype test

The repository now includes a practical **image + prompt → video** prototype using the open-source **Wan2.2 TI2V-5B** base model. Wan2.2 officially supports text-to-video and image-to-video; its authors document the 5B hybrid model as a 720P/24fps model that can run on a consumer GPU such as an RTX 4090, with the official example using offloading. citeturn2search1

This does **not** mean BISWA AI has already trained its own model. The current purpose is to generate a real test clip, inspect quality, and use that result to plan BISWA-specific fine-tuning.

## Try the first video

Open:

- `notebooks/BISWA_AI_Wan22_Test.ipynb`

Run it in a GPU notebook environment, upload one image, write a motion prompt, and generate `biswa_ai_test.mp4`.

The notebook downloads the official Wan2.2 source/checkpoint and runs a short test. GPU availability is the only part that cannot be guaranteed as free.

## Project direction

1. Prototype image-to-video generation.
2. Evaluate face, hands, body motion, identity consistency and prompt following.
3. Prepare a properly licensed training dataset.
4. Fine-tune/LoRA the chosen base model for BISWA-specific behavior.
5. Evaluate checkpoints.
6. Release BISWA model weights according to the chosen model/data licenses.

## Privacy

The project is designed around temporary processing. User media should not be committed to GitHub or stored permanently by the model project.

## License

Wan2.2 is released under Apache 2.0 according to the official model repository. Any BISWA-specific weights/dataset must have their own license and usage terms checked before distribution or commercial use. citeturn0search0turn2search1

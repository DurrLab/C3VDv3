# Depth-conditioned sim2real colonoscopy video generation

This directory contains the **inference** code for turning ground-truth depth
sequences into photorealistic colonoscopy video, built on top of
[LTX-2](https://github.com/Lightricks/LTX-2).

Given a per-frame depth sequence (for example, the ground-truth depth rendered
by the C3VD registration/rendering pipeline in this repository) and a single
real reference frame used as an appearance anchor, the pipeline generates a
video whose geometry follows the depth signal while its appearance matches real
colonoscopy imagery.

## Setup

The code is distributed as an *overlay* on top of an unmodified upstream LTX-2
checkout. `setup_overlay.sh` clones LTX-2 at the pinned commit and copies the
modified files from `ltx_overlay/` into place:

```bash
cd sim2real
bash setup_overlay.sh
cd LTX-2 && uv sync
```

This produces a working `LTX-2/` tree next to the launcher scripts.

## Inputs

- **Base checkpoint** — an LTX-2 colonoscopy base model (`.safetensors`).
- **Depth IC-LoRA** — directory of `lora_weights_step_*.safetensors`.
- **Gemma text encoder** — the LTX-2 Gemma encoder directory.
- **Clips JSON** — frame-list metadata describing each clip's RGB and depth
  frame directories.
- **Reference images** — real frames used as the first-frame appearance anchor,
  plus a `first_frame_mapping.json` pairing each clip with its reference
  image(s).

Depth frames are read directly as per-frame PNGs. Optional depth
preprocessing (spatial filtering, inverse-depth percentile stretch, gamma) is
available in `scripts/prepare_validation_inputs_depth.py`. The trained sim-to-real LoRA weights will be released after the review period.

### Input file formats

`CLIPS_JSON` is a list of clip objects. Each object lists the RGB and depth
frame files for one clip (RGB and depth must have equal length):

```json
[
  {
    "name": "clip_0001",
    "rgb_dir": "/data/clip_0001/rgb",
    "rgb_frames": ["0000.png", "0001.png", "..."],
    "depth_dir": "/data/clip_0001/depth",
    "depth_frames": ["0000.png", "0001.png", "..."],
    "caption": "Real Colonoscopy Image, White Light Imaging",
    "domain": "synthetic"
  }
]
```

`FIRST_FRAME_MAPPING` pairs each clip with the reference image(s) used as the
first-frame appearance anchor. `reference_image` may be a single filename or a
list, resolved relative to `REFERENCE_IMAGES_DIR`:

```json
[
  { "video_name": "clip_0001", "reference_image": "ref_a.png" }
]
```

## Running inference

Set the required environment variables and launch:

```bash
export BASE_MODEL_PATH=/path/to/base_model.safetensors
export TEXT_ENCODER_PATH=/path/to/gemma
export DEPTH_LORA_DIR=/path/to/depth_ic_lora
export CLIPS_JSON=/path/to/clips.json
export FIRST_FRAME_MAPPING=/path/to/first_frame_mapping.json
export REFERENCE_IMAGES_DIR=/path/to/reference_images
export OUTPUT_ROOT=/path/to/outputs

bash run/inference_depth_ic.sh
```

The launcher stages inputs, caches the text-prompt embeddings, and then runs
`inference_edge_ic.py` for every (clip × reference image × seed) tuple, sharded
across the visible GPUs. Generated MP4s and frame dumps are written under
`OUTPUT_ROOT`. See the comment header of `run/inference_depth_ic.sh` for the
full list of optional knobs (resolution, frame count, guidance/STG settings,
depth preprocessing parameters, etc.).

## Layout

```
sim2real/
├── setup_overlay.sh        # clone upstream LTX-2 + apply overlay
├── run/
│   └── inference_depth_ic.sh
├── scripts/
│   ├── prepare_validation_inputs_depth.py   # stage depth/reference inputs
│   └── precompute_validation_prompts.py     # cache Gemma prompt embeddings
└── ltx_overlay/            # modified LTX-2 source files (applied by setup)
    └── packages/...
```

## License

The contents of this `sim2real/` directory are distributed under the **LTX-2
Community License** (see `LICENSE`), as they derive from LTX-2. See `NOTICE`
for the list of modified upstream files. This license applies only to this
directory; the rest of the C3VDv3 repository is covered by its own license.

#!/usr/bin/env python3
# ruff: noqa: T201
"""
Per-frame IC-LoRA inference for depth-conditioned video generation.

Key property: --reference-frames-dir loads a sequence of PNG/JPG frames
directly into the VAE encoder, with NO MP4 round-trip. This preserves the
reference depth signal at full fidelity; an MP4 round-trip would apply H.264
chroma subsampling and 8-bit quantization and crush the fine depth gradients
the conditioning relies on.

Additional features beyond upstream inference.py:
- --cached-condition-path: pre-encoded Gemma + connector embedding from disk,
  so per-clip jobs do not pay the Gemma load cost.
- --reference-resize-mode / --condition-image-resize-mode: 'crop' (legacy
  aspect-preserving + center crop) or 'stretch' (direct bilinear resize).
- --reference-strength, --condition-image-strength: partial denoise the
  reference / first-frame tokens.
- --source-frames-dir, --output-frames-dir,
  --include-reference-in-output: side-by-side visualization and raw PNG
  output dumping.
- Long-video sliding-window mode (--total-frames > --num-frames): chunks
  stitched with linear-ramp crossfade; chunk i>0 is i2v-anchored on a frame
  copied from chunk i-1's interior, so trajectory does not drift at chunk
  boundaries.

Usage (Depth IC I2V, single chunk):
    python scripts/inference_edge_ic.py --checkpoint base_model.safetensors \\
        --text-encoder-path path/to/gemma --lora-path depth_ic_lora.safetensors \\
        --prompt "Real Colonoscopy Image, White Light Imaging" \\
        --condition-image first_frame.jpg \\
        --reference-frames-dir /path/to/depth_frames/{clip} \\
        --reference-resize-mode stretch \\
        --include-reference-in-output --output out.mp4
"""

import argparse
import gc
import math
import re
from dataclasses import replace
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from ltx_trainer.model_loader import load_embeddings_processor, load_model, load_text_encoder
from ltx_trainer.progress import StandaloneSamplingProgress
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video


_FRAME_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def count_image_frames(frames_dir: str) -> int:
    """Count sortable image files in a directory; 0 if not a directory."""
    path = Path(frames_dir)
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in _FRAME_EXTS)


def load_reference_frames_dir(frames_dir: str, max_frames: int) -> torch.Tensor:
    """Load an ordered sequence of image frames from a directory as [F, C, H, W] in [0, 1].

    Matches the shape/range contract of read_video() so downstream code is unchanged.
    Files are sorted by name; the first `max_frames` are loaded. All frames must
    share the same resolution.
    """
    path = Path(frames_dir)
    if not path.is_dir():
        raise ValueError(f"--reference-frames-dir is not a directory: {frames_dir}")

    frame_paths = sorted(p for p in path.iterdir() if p.suffix.lower() in _FRAME_EXTS)
    if not frame_paths:
        raise ValueError(f"No image frames found under {frames_dir} (looked for {_FRAME_EXTS})")

    frame_paths = frame_paths[:max_frames]
    tensors = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        tensors.append(to_tensor(img))  # [C, H, W] in [0, 1]

    shapes = {t.shape for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Reference frames have mismatched shapes under {frames_dir}: {shapes}")

    return torch.stack(tensors, dim=0)  # [F, C, H, W]


def load_reference_frames_range(frames_dir: str, start: int, count: int) -> torch.Tensor:
    """Load a sub-range of sorted image frames as [F, C, H, W] in [0, 1].

    Same shape/range contract as load_reference_frames_dir, but slices the
    sorted file list to [start : start + count]. Errors if fewer than
    start + count frames are available, so callers fail fast rather than
    silently truncating a chunk.
    """
    path = Path(frames_dir)
    if not path.is_dir():
        raise ValueError(f"--reference-frames-dir is not a directory: {frames_dir}")

    frame_paths = sorted(p for p in path.iterdir() if p.suffix.lower() in _FRAME_EXTS)
    if len(frame_paths) < start + count:
        raise ValueError(
            f"Reference frames dir {frames_dir} has only {len(frame_paths)} frames; "
            f"need at least {start + count} for chunk [{start}:{start + count}]"
        )

    frame_paths = frame_paths[start : start + count]
    tensors = [to_tensor(Image.open(fp).convert("RGB")) for fp in frame_paths]
    shapes = {t.shape for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Reference frames have mismatched shapes under {frames_dir}: {shapes}")
    return torch.stack(tensors, dim=0)


def compute_chunks(total: int, chunk: int, overlap: int) -> list[int]:
    """Return chunk start indices that tile [0, total) with `chunk`-sized windows.

    Uses stride = chunk - overlap. The last start is forced to total - chunk so
    the run terminates exactly at frame total - 1; this can make the final
    overlap larger than `overlap` when (total - chunk) is not divisible by
    stride. Crossfade weights sum to 1 per frame, so asymmetric tails are fine.
    """
    if total < chunk:
        raise ValueError(f"total ({total}) must be >= chunk ({chunk})")
    stride = chunk - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < chunk ({chunk})")
    n = math.ceil(max(total - chunk, 0) / stride) + 1
    return [i * stride for i in range(n - 1)] + [total - chunk]


def stitch_chunks(
    chunks: list[torch.Tensor], starts: list[int], total: int, chunk_size: int
) -> torch.Tensor:
    """Linear-ramp crossfade in pixel space [0, 1]; output [C, total, H, W].

    Per-frame weights sum to 1 everywhere — including asymmetric overlaps —
    because each overlap pair is (linspace 1->0) + (linspace 0->1), summing to
    1 at every interior position. At the very first overlap frame the
    incoming chunk contributes 0 (anchor frame is discarded in favor of the
    prior chunk, which is already locked to the same global frame); at the
    last it contributes 1.
    """
    if not chunks:
        raise ValueError("stitch_chunks called with no chunks")
    c0 = chunks[0]
    C, _, H, W = c0.shape
    out = torch.zeros((C, total, H, W), dtype=c0.dtype)
    weight = torch.zeros(total, dtype=c0.dtype)
    for i, (s, c) in enumerate(zip(starts, chunks)):
        a = torch.ones(chunk_size, dtype=c0.dtype)
        if i > 0:
            ov = (starts[i - 1] + chunk_size) - s
            if ov > 1:
                a[:ov] = torch.linspace(0.0, 1.0, ov, dtype=c0.dtype)
            elif ov == 1:
                a[0] = 0.0
        if i < len(chunks) - 1:
            ov_next = (s + chunk_size) - starts[i + 1]
            if ov_next > 1:
                a[-ov_next:] = torch.linspace(1.0, 0.0, ov_next, dtype=c0.dtype)
            elif ov_next == 1:
                a[-1] = 0.0
        out[:, s : s + chunk_size] += a[None, :, None, None] * c
        weight[s : s + chunk_size] += a
    return out / weight[None, :, None, None].clamp(min=1e-6)


def load_image(image_path: str) -> torch.Tensor:
    """Load an image and convert to tensor [C, H, W] in [0, 1]."""
    image = open_image_as_srgb(image_path)
    transform = transforms.ToTensor()
    return transform(image)


def load_cached_condition(condition_path: str | Path) -> CachedPromptEmbeddings:
    """Load connected prompt embeddings saved by precompute_validation_prompts.py.

    Optionally reads negative embeddings under keys ``video_neg_embeds`` /
    ``audio_neg_embeds``; when present the returned embeddings support CFG
    (guidance_scale != 1.0) without a live Gemma encode.
    """
    payload = torch.load(condition_path, map_location="cpu")
    required = {"video_prompt_embeds", "audio_prompt_embeds"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Condition file is missing keys: {sorted(missing)}")

    cached = CachedPromptEmbeddings(
        video_context_positive=payload["video_prompt_embeds"].unsqueeze(0),
        audio_context_positive=payload["audio_prompt_embeds"].unsqueeze(0),
    )
    if "video_neg_embeds" in payload:
        cached.video_context_negative = payload["video_neg_embeds"].unsqueeze(0)
    if "audio_neg_embeds" in payload:
        cached.audio_context_negative = payload["audio_neg_embeds"].unsqueeze(0)
    return cached


def extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Extract target module names from LoRA checkpoint keys.
    LoRA keys follow the pattern (after removing "diffusion_model." prefix):
    - transformer_blocks.0.attn1.to_k.lora_A.weight
    - transformer_blocks.0.ff.net.0.proj.lora_B.weight
    This extracts the full module path like "transformer_blocks.0.attn1.to_k".
    Using full paths is more robust than partial patterns.
    """
    target_modules = set()
    # Pattern to extract everything before .lora_A or .lora_B
    pattern = re.compile(r"(.+)\.lora_[AB]\.")

    for key in state_dict:
        match = pattern.match(key)
        if match:
            module_path = match.group(1)
            target_modules.add(module_path)

    return sorted(target_modules)


def load_lora_weights(transformer: torch.nn.Module, lora_path: str | Path, lora_scale: float = 1.0) -> torch.nn.Module:
    """Load LoRA weights into the transformer model.
    The LoRA rank and target modules are automatically detected from the checkpoint.
    Alpha is set equal to rank (standard practice for inference).
    Args:
        transformer: The base transformer model
        lora_path: Path to the LoRA weights (.safetensors)
    Returns:
        The transformer model with LoRA weights applied
    """
    print(f"Loading LoRA weights from {lora_path}...")

    # Load the LoRA state dict
    state_dict = load_file(str(lora_path))

    # Remove "diffusion_model." prefix (ComfyUI-compatible format)
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

    # Extract target modules from the checkpoint
    target_modules = extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"Could not extract target modules from LoRA checkpoint: {lora_path}")
    print(f"  Detected {len(target_modules)} target modules")

    # Auto-detect rank from the first lora_A weight shape
    lora_rank = None
    for key, value in state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
            break
    if lora_rank is None:
        raise ValueError("Could not auto-detect LoRA rank from weights")
    print(f"  LoRA rank: {lora_rank}")

    # Create LoRA config and wrap the model
    # Alpha = rank is standard for inference (maintains the trained scale)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )

    # Wrap the transformer with PEFT to add LoRA layers
    transformer = get_peft_model(transformer, lora_config)

    # Load the LoRA weights
    base_model = transformer.get_base_model()
    set_peft_model_state_dict(base_model, state_dict)

    # Optionally scale LoRA influence at inference time.
    if lora_scale != 1.0:
        scaled_layers = 0
        for module in transformer.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict):
                for adapter_name in list(scaling.keys()):
                    scaling[adapter_name] = scaling[adapter_name] * lora_scale
                    scaled_layers += 1
        print(f"  Applied LoRA scale: {lora_scale} ({scaled_layers} adapter scales updated)")

    print("✓ LoRA weights loaded successfully")
    return transformer


def main() -> None:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="LTX Video/Audio Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        required=True,
        help="Path to Gemma text encoder directory",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights (.safetensors)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for LoRA strength during inference",
    )
    parser.add_argument(
        "--cached-condition-path",
        type=str,
        default=None,
        help="Path to a precomputed connected condition .pt file to skip Gemma prompt encoding",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=544,
        help="Video height (must be divisible by 32)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Video width (must be divisible by 32)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=97,
        help="Number of video frames per chunk (must be k*8 + 1)",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help=(
            "Total output video length. Default = --num-frames (single-chunk mode, "
            "behavior unchanged from before). When > --num-frames, runs sliding-window "
            "chunked generation and crossfades the overlapping regions in pixel space."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=8,
        help=(
            "Frame overlap between adjacent chunks in long-video mode. Stride = "
            "num_frames - chunk_overlap. The final chunk may overlap more than this "
            "value so the run lands exactly on total_frames."
        ),
    )
    parser.add_argument(
        "--chunk-seed-stride",
        type=int,
        default=0,
        help=(
            "Per-chunk seed offset; chunk i uses seed + i * stride. Default 0 reuses the "
            "same seed for every chunk."
        ),
    )
    parser.add_argument(
        "--chunk-debug-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to dump per-chunk MP4 + PNG frames before stitching, "
            "for inspecting raw chunk outputs in long-video mode."
        ),
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Video frame rate",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (CFG)",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=1.0,
        help="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. Default: 1.0",
    )
    parser.add_argument(
        "--stg-blocks",
        type=int,
        nargs="*",
        default=[29],
        help="Which transformer blocks to perturb for STG. Default: 29 (single block).",
    )
    parser.add_argument(
        "--stg-mode",
        type=str,
        default="stg_av",
        choices=["stg_av", "stg_v"],
        help="STG mode: 'stg_av' perturbs both audio and video, 'stg_v' perturbs video only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Conditioning arguments
    parser.add_argument(
        "--condition-image",
        type=str,
        default=None,
        help="Path to conditioning image for image-to-video generation",
    )
    parser.add_argument(
        "--reference-video",
        type=str,
        default=None,
        help="Path to reference video for video-to-video generation (IC-LoRA style)",
    )
    parser.add_argument(
        "--reference-frames-dir",
        type=str,
        default=None,
        help=(
            "Path to a directory of PNG/JPG frames used as the reference (IC-LoRA style). "
            "Frames are sorted by filename and loaded directly (no MP4 encode/decode). "
            "Mutually exclusive with --reference-video."
        ),
    )
    parser.add_argument(
        "--include-reference-in-output",
        action="store_true",
        help="Include reference video side-by-side with generated output (only for V2V)",
    )
    parser.add_argument(
        "--source-frames-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of source-clip RGB frames (PNG/JPG, sorted by name) to "
            "concatenate as the leftmost panel of the saved video, alongside the reference "
            "and generated panels. Useful for visualizing the original RGB clip from which "
            "the depth/edge condition was derived."
        ),
    )
    parser.add_argument(
        "--reference-strength",
        type=float,
        default=1.0,
        help=(
            "IC-LoRA reference conditioning strength in [0, 1]. "
            "1.0 = full (reference kept clean, current behavior), "
            "0.0 = none (reference denoised with target). "
            "Matches LTX docs' depth attention_strength / video-conditioning strength."
        ),
    )
    parser.add_argument(
        "--reference-resize-mode",
        type=str,
        choices=["crop", "stretch"],
        default="crop",
        help=(
            "How to resize reference frames before IC-LoRA conditioning. "
            "'crop' preserves aspect ratio and center-crops (default/current behavior); "
            "'stretch' directly resizes to the output size with no crop."
        ),
    )
    parser.add_argument(
        "--condition-image-resize-mode",
        type=str,
        choices=["crop", "stretch"],
        default="crop",
        help=(
            "How to resize the conditioning image (--condition-image) before VAE encoding. "
            "'crop' = aspect-preserving resize-to-cover + center crop (default/legacy); "
            "'stretch' = direct bilinear resize to (height, width), no crop."
        ),
    )
    parser.add_argument(
        "--condition-image-strength",
        type=float,
        default=1.0,
        help=(
            "First-frame (I2V) conditioning strength in [0, 1]. "
            "1.0 = fully frozen (current behavior), <1.0 = partial denoising of the "
            "conditioning tokens."
        ),
    )

    # Audio arguments
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio generation (by default, audio is generated alongside video)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path (.mp4)",
    )
    parser.add_argument(
        "--output-frames-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for generated RGB frames. Saves raw generated frames "
            "as 0000.png, 0001.png, ... before any side-by-side visualization panels "
            "are added to the MP4 output."
        ),
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default=None,
        help="Output audio path (.wav, optional - if not provided, audio will be embedded in video)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--load-text-encoder-in-8bit",
        action="store_true",
        help="Load Gemma in 8-bit precision for lower-memory prompt encoding",
    )

    args = parser.parse_args()

    # Validate conditioning arguments
    if args.reference_video is not None and args.reference_frames_dir is not None:
        parser.error("--reference-video and --reference-frames-dir are mutually exclusive")
    has_reference = args.reference_video is not None or args.reference_frames_dir is not None
    if args.include_reference_in_output and not has_reference:
        parser.error(
            "--include-reference-in-output requires --reference-video or --reference-frames-dir"
        )
    if not (0.0 <= args.reference_strength <= 1.0):
        parser.error(
            f"--reference-strength must be in [0.0, 1.0], got {args.reference_strength}"
        )
    if not (0.0 <= args.condition_image_strength <= 1.0):
        parser.error(
            f"--condition-image-strength must be in [0.0, 1.0], got {args.condition_image_strength}"
        )

    # Long-video / chunked-mode validation
    if args.total_frames is None:
        args.total_frames = args.num_frames
    if args.total_frames < args.num_frames:
        parser.error(
            f"--total-frames ({args.total_frames}) must be >= --num-frames ({args.num_frames})"
        )
    if not (0 <= args.chunk_overlap < args.num_frames):
        parser.error(
            f"--chunk-overlap must be in [0, {args.num_frames}), got {args.chunk_overlap}"
        )
    long_mode = args.total_frames > args.num_frames
    if long_mode:
        if args.reference_frames_dir is None:
            parser.error(
                "Long-video mode (--total-frames > --num-frames) requires "
                "--reference-frames-dir; --reference-video is not supported because "
                "the MP4 read path slices to --num-frames and cannot serve later chunks."
            )
        if args.condition_image is None:
            parser.error(
                "Long-video mode requires --condition-image (I2V anchor). "
                "T2V long mode is not implemented."
            )
        if not args.skip_audio:
            parser.error(
                "Long-video mode requires --skip-audio. Chunked audio reassembly is not "
                "implemented; pass --skip-audio explicitly."
            )
        # Preflight: refuse before model load if depth dir is too short, so the
        # user does not pay for prompt encoding + LoRA load + multiple chunk
        # generations only to die on the last chunk.
        n_avail = count_image_frames(args.reference_frames_dir)
        if n_avail < args.total_frames:
            parser.error(
                f"--reference-frames-dir has {n_avail} frames but --total-frames is "
                f"{args.total_frames}; long-video mode requires at least total_frames "
                f"depth frames in the directory."
            )
        # Reference visualization (--include-reference-in-output) feeds the full
        # sequence through ValidationSampler._preprocess_reference_video, which
        # trims to k*8+1 internally. Reject totals that would silently misalign.
        if args.include_reference_in_output and (args.total_frames - 1) % 8 != 0:
            parser.error(
                f"--include-reference-in-output in long-video mode requires "
                f"--total-frames satisfying (total_frames - 1) % 8 == 0 "
                f"(reference preprocessing trims to k*8+1). Got {args.total_frames}."
            )
        # Source-frames panel guard: silent truncation/padding would produce a
        # misleading visual comparison.
        if args.source_frames_dir is not None:
            n_src = count_image_frames(args.source_frames_dir)
            if n_src < args.total_frames:
                parser.error(
                    f"--source-frames-dir has {n_src} frames but --total-frames is "
                    f"{args.total_frames}; need at least total_frames source frames "
                    f"to render the side-by-side panel without padding."
                )

    # Validate arguments
    generate_audio = not args.skip_audio

    print("=" * 80)
    print("LTX Video/Audio Generation")
    print("=" * 80)

    # Determine if we need VAE encoder (for image or video conditioning)
    need_vae_encoder = (
        args.condition_image is not None
        or args.reference_video is not None
        or args.reference_frames_dir is not None
    )

    use_cached_condition = args.cached_condition_path is not None

    components = load_model(
        checkpoint_path=args.checkpoint,
        device="cpu",  # Load to CPU first, sampler will move to device as needed
        dtype=torch.bfloat16,
        with_video_vae_encoder=need_vae_encoder,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=generate_audio,
        with_vocoder=generate_audio,
        with_text_encoder=False,
    )
    if use_cached_condition:
        cached_embeddings = load_cached_condition(args.cached_condition_path)
        if args.guidance_scale != 1.0 and cached_embeddings.video_context_negative is None:
            raise ValueError(
                "--cached-condition-path with --guidance-scale != 1.0 requires "
                "negative embeddings (video_neg_embeds / audio_neg_embeds) in the "
                "cache file. Regenerate the cache with precompute_validation_prompts.py."
            )
        embeddings_processor = None
        components.text_encoder = None
    else:
        components.text_encoder = load_text_encoder(
            gemma_model_path=args.text_encoder_path,
            device="cpu",
            dtype=torch.bfloat16,
            load_in_8bit=args.load_text_encoder_in_8bit,
        )
        embeddings_processor = load_embeddings_processor(
            checkpoint_path=args.checkpoint,
            device="cpu",
            dtype=torch.bfloat16,
        )

        prompt_device = torch.device(args.device)

        # Pre-compute prompt embeddings before sampling so we can free Gemma prior
        # to moving the video generation stack onto the GPU.
        components.text_encoder.to(prompt_device)
        embeddings_processor.to(prompt_device)
        with torch.inference_mode():
            pos_hs, pos_mask = components.text_encoder.encode(args.prompt)
            pos_out = embeddings_processor.process_hidden_states(pos_hs, pos_mask)

            cached_embeddings = CachedPromptEmbeddings(
                video_context_positive=pos_out.video_encoding.cpu(),
                audio_context_positive=pos_out.audio_encoding.cpu(),
            )

            if args.guidance_scale != 1.0:
                neg_hs, neg_mask = components.text_encoder.encode(args.negative_prompt)
                neg_out = embeddings_processor.process_hidden_states(neg_hs, neg_mask)
                cached_embeddings.video_context_negative = neg_out.video_encoding.cpu()
                cached_embeddings.audio_context_negative = (
                    neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
                )

        components.text_encoder.to("cpu")
        embeddings_processor.to("cpu")
        if prompt_device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        components.text_encoder = None
        embeddings_processor = None

    # Apply LoRA weights if provided
    transformer = components.transformer
    if args.lora_path is not None:
        transformer = load_lora_weights(transformer, args.lora_path, lora_scale=args.lora_scale)

    # Load conditioning image if provided
    condition_image = None
    if args.condition_image:
        print(f"Loading conditioning image from {args.condition_image}...")
        condition_image = load_image(args.condition_image)

    # Load reference video if provided (either MP4 or PNG frame directory)
    reference_video = None
    reference_source = None  # for logging
    if args.reference_video:
        print(f"Loading reference video from {args.reference_video}...")
        # In long-video mode --reference-video is rejected upstream, so total_frames
        # always equals num_frames here and behavior is unchanged.
        reference_video, ref_fps = read_video(args.reference_video, max_frames=args.num_frames)
        print(f"  Loaded {reference_video.shape[0]} frames @ {ref_fps:.1f} fps")
        reference_source = f"Video ({args.reference_video})"
    elif args.reference_frames_dir:
        # Single-chunk: preload num_frames worth (= existing behavior).
        # Long mode: preload the full sequence ONLY when --include-reference-in-output
        # asks for it; otherwise the chunk loop reads its own slices via
        # load_reference_frames_range and we avoid a ~total_frames-sized buffer in RAM.
        if not long_mode:
            print(f"Loading reference frames from directory {args.reference_frames_dir}...")
            reference_video = load_reference_frames_dir(
                args.reference_frames_dir, max_frames=args.num_frames
            )
            print(
                f"  Loaded {reference_video.shape[0]} frames "
                f"({reference_video.shape[2]}x{reference_video.shape[3]}) "
                f"directly from PNG/JPG (no MP4 intermediate)"
            )
        elif args.include_reference_in_output:
            print(
                f"Loading full {args.total_frames}-frame reference for visualization "
                f"from {args.reference_frames_dir}..."
            )
            reference_video = load_reference_frames_dir(
                args.reference_frames_dir, max_frames=args.total_frames
            )
            print(
                f"  Loaded {reference_video.shape[0]} frames "
                f"({reference_video.shape[2]}x{reference_video.shape[3]})"
            )
        else:
            # Long mode without visualization: skip the preload; chunk loop will
            # read slices on demand. Leave reference_video as None.
            print(
                f"Long-video mode: skipping full reference preload "
                f"(visualization disabled). Chunks will read slices from "
                f"{args.reference_frames_dir} on demand."
            )
        reference_source = f"Frames dir ({args.reference_frames_dir})"

    # has_reference reflects whether a reference is in use, not whether it's
    # preloaded into memory. In long mode without --include-reference-in-output,
    # reference_video is None but the chunk loop still consumes per-chunk slices,
    # so the run is genuinely V2V.
    has_reference = (
        reference_video is not None
        or args.reference_video is not None
        or args.reference_frames_dir is not None
    )

    # Determine generation mode
    if has_reference and args.condition_image is not None:
        mode = "Video-to-Video + Image Conditioning (V2V+I2V)"
    elif has_reference:
        mode = "Video-to-Video (V2V)"
    elif args.condition_image is not None:
        mode = "Image-to-Video (I2V)"
    else:
        mode = "Text-to-Video (T2V)"

    print("\n" + "=" * 80)
    print("Generation Parameters")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.frame_rate} fps")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"CFG scale: {args.guidance_scale}")
    if args.stg_scale > 0:
        blocks_str = args.stg_blocks if args.stg_blocks else "all"
        print(f"STG scale: {args.stg_scale} (mode: {args.stg_mode}, blocks: {blocks_str})")
    else:
        print("STG: disabled")
    print(f"Seed: {args.seed}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    if condition_image is not None:
        print(f"Conditioning: Image ({args.condition_image})")
    if reference_video is not None:
        print(f"Reference: {reference_source}")
        print(f"  Resize mode: {args.reference_resize_mode}")
        if args.include_reference_in_output:
            print("  → Will include reference side-by-side in output")
    if generate_audio:
        video_duration = args.num_frames / args.frame_rate
        print(f"Audio: Enabled (duration will match video: {video_duration:.2f}s)")
    print("=" * 80)

    print(f"\nGenerating {'video + audio' if generate_audio else 'video'}...")

    # Create generation config (used directly in single-chunk mode; in long mode
    # this serves as a template that the chunk loop overrides per-chunk).
    gen_config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        condition_image=condition_image,
        reference_video=reference_video,
        reference_resize_mode=args.reference_resize_mode,
        condition_image_resize_mode=args.condition_image_resize_mode,
        generate_audio=generate_audio,
        include_reference_in_output=False,
        cached_embeddings=cached_embeddings,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        stg_mode=args.stg_mode,
        reference_strength=args.reference_strength,
        condition_image_strength=args.condition_image_strength,
    )

    if not long_mode:
        # ── Single-chunk path: structurally unchanged from pre-long-mode code. ──
        with StandaloneSamplingProgress(num_steps=args.num_inference_steps) as progress:
            sampler = ValidationSampler(
                transformer=transformer,
                vae_decoder=components.video_vae_decoder,
                vae_encoder=components.video_vae_encoder,
                text_encoder=components.text_encoder,
                embeddings_processor=embeddings_processor,
                audio_decoder=components.audio_vae_decoder if generate_audio else None,
                vocoder=components.vocoder if generate_audio else None,
                sampling_context=progress,
            )
            video, audio = sampler.generate(
                config=gen_config,
                device=args.device,
            )
    else:
        # ── Long-video sliding-window mode ──
        starts = compute_chunks(args.total_frames, args.num_frames, args.chunk_overlap)
        print()
        print("=" * 80)
        print("Long-video sliding-window generation")
        print("=" * 80)
        print(f"  total_frames  = {args.total_frames}")
        print(f"  chunk_size    = {args.num_frames}")
        print(f"  chunk_overlap = {args.chunk_overlap} (last chunk may overlap more)")
        print(f"  num_chunks    = {len(starts)}")
        for i, s in enumerate(starts):
            ov_prev = (starts[i - 1] + args.num_frames - s) if i > 0 else 0
            print(
                f"    chunk {i}: frames [{s}..{s + args.num_frames - 1}] "
                f"(overlap with prev = {ov_prev})"
            )
        print("=" * 80)

        # Build the sampler once; rebind sampling_context per chunk for clean
        # progress bars. Audio is disabled in long mode (validated upstream).
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=components.text_encoder,
            embeddings_processor=embeddings_processor,
            audio_decoder=None,
            vocoder=None,
            sampling_context=None,
        )

        chunk_videos: list[torch.Tensor] = []  # each [C, num_frames, H, W] on CPU in [0,1]
        for i, s in enumerate(starts):
            depth_slice = load_reference_frames_range(
                args.reference_frames_dir, s, args.num_frames
            )
            if i == 0:
                cond = condition_image
            else:
                # Anchor chunk i at the previous chunk's frame whose global index
                # equals the new chunk's start. Using the previous chunk's last
                # frame would silently shift the RGB trajectory by the overlap.
                prev_local_idx = starts[i] - starts[i - 1]
                cond = chunk_videos[-1][:, prev_local_idx].clone().contiguous()

            chunk_cfg = replace(
                gen_config,
                seed=args.seed + i * args.chunk_seed_stride,
                condition_image=cond,
                reference_video=depth_slice,
                num_frames=args.num_frames,
                generate_audio=False,
                include_reference_in_output=False,
            )

            print(f"\n[chunk {i + 1}/{len(starts)}] frames [{s}..{s + args.num_frames - 1}] "
                  f"seed={chunk_cfg.seed}")
            with StandaloneSamplingProgress(num_steps=args.num_inference_steps) as progress:
                sampler._sampling_context = progress
                chunk_video, _ = sampler.generate(config=chunk_cfg, device=args.device)
            chunk_cpu = chunk_video.detach().to("cpu", dtype=torch.float32)
            chunk_videos.append(chunk_cpu)

            if args.chunk_debug_dir is not None:
                debug_root = Path(args.chunk_debug_dir)
                debug_root.mkdir(parents=True, exist_ok=True)
                chunk_frames_dir = debug_root / f"chunk_{i:02d}"
                chunk_frames_dir.mkdir(parents=True, exist_ok=True)
                for old_frame in chunk_frames_dir.glob("*.png"):
                    old_frame.unlink()
                for fi in range(chunk_cpu.shape[1]):
                    save_image(chunk_cpu[:, fi], chunk_frames_dir / f"{fi:04d}.png")
                save_video(
                    video_tensor=chunk_cpu,
                    output_path=debug_root / f"chunk_{i:02d}.mp4",
                    fps=args.frame_rate,
                    audio=None,
                    audio_sample_rate=None,
                )
                print(f"  → debug dump: {chunk_frames_dir}/ + {debug_root / f'chunk_{i:02d}.mp4'}")

        # Crossfade in pixel space.
        video = stitch_chunks(chunk_videos, starts, args.total_frames, args.num_frames)
        audio = None

        # Replace gen_config so the visualization tail (--include-reference-in-output)
        # operates on the full reference sequence rather than the last chunk's slice.
        gen_config = replace(gen_config, reference_video=reference_video, num_frames=args.total_frames)
        print(f"\n✓ Stitched {len(starts)} chunks into {video.shape[1]}-frame video")

    if args.output_frames_dir is not None:
        frames_dir = Path(args.output_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frames_dir.glob("*.png"):
            old_frame.unlink()
        for frame_idx in range(video.shape[1]):
            save_image(video[:, frame_idx], frames_dir / f"{frame_idx:04d}.png")
        print(f"✓ Generated RGB frames saved to {frames_dir}")

    # Optionally concatenate original reference video side-by-side for MP4 output
    if args.include_reference_in_output and reference_video is not None:
        ref_video_preprocessed = ValidationSampler._preprocess_reference_video(gen_config)
        ref_video_pixels = ((ref_video_preprocessed[0].cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
        video = ValidationSampler._concatenate_videos_side_by_side(ref_video_pixels, video)

    # Optionally prepend source RGB frames as the leftmost side-by-side panel
    if args.source_frames_dir is not None:
        print(f"Loading source RGB frames from directory {args.source_frames_dir}...")
        source_frames = load_reference_frames_dir(
            args.source_frames_dir, max_frames=args.total_frames
        )  # [F, C, H, W] in [0, 1]
        source_video = source_frames.permute(1, 0, 2, 3).contiguous().to(video.device, video.dtype)
        print(
            f"  Loaded {source_video.shape[1]} source frames "
            f"({source_video.shape[2]}x{source_video.shape[3]})"
        )
        target_height = args.height
        target_width = args.width
        if source_video.shape[2] != target_height or source_video.shape[3] != target_width:
            _, _, h, w = source_video.shape
            source_video = source_video.permute(1, 0, 2, 3)
            source_video = torch.nn.functional.interpolate(
                source_video,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            source_video = source_video.permute(1, 0, 2, 3).contiguous()
            print(
                f"  Resized source RGB panel from {h}x{w} "
                f"to {target_height}x{target_width} (direct resize, no crop)"
            )
        video = ValidationSampler._concatenate_videos_side_by_side(source_video, video)

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get audio sample rate from vocoder if audio was generated
    audio_sample_rate = None
    if audio is not None and components.vocoder is not None:
        audio_sample_rate = components.vocoder.output_sampling_rate

    save_video(
        video_tensor=video,
        output_path=output_path,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
    )
    print(f"✓ Video saved to {args.output}")

    # Save separate audio file if requested
    if audio is not None and args.audio_output is not None:
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "torchaudio is only required when saving a separate audio file via --audio-output. "
                "Install torchaudio or omit --audio-output."
            ) from exc

        audio_output_path = Path(args.audio_output)
        audio_output_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(audio_output_path),
            audio.cpu(),
            sample_rate=audio_sample_rate,
        )
        duration = audio.shape[1] / audio_sample_rate
        print(f"✓ Audio saved: {duration:.2f}s at {audio_sample_rate}Hz")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

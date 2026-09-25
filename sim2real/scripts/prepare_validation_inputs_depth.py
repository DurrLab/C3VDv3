#!/usr/bin/env python3
"""Prepare image-to-video + Depth-IC validation inputs.

Stages validation clips from a clips JSON into a manifest plus per-clip frame
directories used by run/inference_depth_ic.sh. By default, depth frames are
symlinked exactly as stored on disk. For synthetic-domain probing the script
can instead materialize preprocessed PNGs (Gaussian / median / bilateral
filtering, optional inverse-depth disparity stretching, gamma) to soften the
sharp fold gradients of rendered depth before inference.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _caption_style(caption: str) -> str:
    if "narrow band" in caption.lower():
        return "nbi"
    if "white light" in caption.lower():
        return "wli"
    return "other"


def _eligible(clip: dict[str, Any], num_frames: int, domain: str | None) -> bool:
    for key in ("name", "rgb_dir", "rgb_frames", "depth_dir", "depth_frames", "caption"):
        if key not in clip:
            return False
    rgb = clip["rgb_frames"]
    depth = clip["depth_frames"]
    if len(rgb) != len(depth) or len(rgb) < num_frames:
        return False
    if domain is not None and clip.get("domain") != domain:
        return False
    return True


def _relink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _clear_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            _clear_dir(child)
            child.rmdir()


def _resolve_depth_frame(depth_dir: Path, frame_name: str, depth_suffix: str) -> Path | None:
    primary = depth_dir / frame_name
    if primary.exists():
        return primary
    if depth_suffix:
        p = Path(frame_name)
        alt = depth_dir / f"{p.stem}{depth_suffix}{p.suffix}"
        if alt.exists():
            return alt
    return None


def _to_grayscale_array(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., :3].astype(np.float32).mean(axis=2)
    return arr.astype(np.float32, copy=False)


def _scale_from_256(height: int, width: int) -> float:
    return max(1e-6, min(height, width) / 256.0)


def _bilateral_filter(
    arr: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    if diameter <= 0:
        raise ValueError(f"--bilateral-diameter must be > 0, got {diameter}")
    if sigma_color <= 0.0:
        raise ValueError(
            f"--bilateral-sigma-color must be > 0.0 for bilateral filtering, got {sigma_color}"
        )
    if sigma_space <= 0.0:
        raise ValueError(
            f"--bilateral-sigma-space-256 must produce sigma_space > 0.0, got {sigma_space}"
        )

    radius = diameter // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma_space * sigma_space)).astype(
        np.float32
    )

    padded = np.pad(arr, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (diameter, diameter))
    center = arr[..., None, None]
    range_weights = np.exp(
        -0.5 * ((windows - center) / sigma_color) * ((windows - center) / sigma_color)
    ).astype(np.float32)
    weights = range_weights * spatial[None, None, :, :]
    weighted = weights * windows
    denom = np.maximum(weights.sum(axis=(-2, -1)), 1e-8)
    return weighted.sum(axis=(-2, -1)) / denom


def _apply_filter_pipeline(
    arr: np.ndarray,
    filter_type: str,
    gaussian_sigma_256: float,
    median_kernel_size: int,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space_256: float,
) -> np.ndarray:
    h, w = arr.shape[-2:]
    scale = _scale_from_256(h, w)
    gaussian_sigma = gaussian_sigma_256 * scale

    if filter_type == "gaussian":
        image = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8))
        image = image.filter(ImageFilter.GaussianBlur(radius=gaussian_sigma))
        return np.asarray(image, dtype=np.float32) / 255.0

    if filter_type == "median_gaussian":
        if median_kernel_size < 3 or median_kernel_size % 2 == 0:
            raise ValueError(
                f"--median-kernel-size must be odd and >= 3 for median_gaussian, got {median_kernel_size}"
            )
        image = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8))
        image = image.filter(ImageFilter.MedianFilter(size=median_kernel_size))
        image = image.filter(ImageFilter.GaussianBlur(radius=gaussian_sigma))
        return np.asarray(image, dtype=np.float32) / 255.0

    if filter_type == "bilateral":
        sigma_space = bilateral_sigma_space_256 * scale
        return _bilateral_filter(
            arr.astype(np.float32),
            diameter=bilateral_diameter,
            sigma_color=bilateral_sigma_color,
            sigma_space=sigma_space,
        )

    raise ValueError(f"Unsupported --depth-filter-type: {filter_type}")


def _save_preprocessed_depth(
    src: Path,
    dst: Path,
    filter_type: str,
    gaussian_sigma_256: float,
    median_kernel_size: int,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space_256: float,
    depth_gamma: float,
) -> None:
    arr = _to_grayscale_array(src)

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    filtered = _apply_filter_pipeline(
        arr,
        filter_type=filter_type,
        gaussian_sigma_256=gaussian_sigma_256,
        median_kernel_size=median_kernel_size,
        bilateral_diameter=bilateral_diameter,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space_256=bilateral_sigma_space_256,
    )

    min_val = float(filtered.min())
    max_val = float(filtered.max())
    if max_val > min_val:
        filtered = (filtered - min_val) / (max_val - min_val)
    else:
        filtered = np.zeros_like(filtered, dtype=np.float32)

    if depth_gamma != 1.0:
        filtered = np.power(np.clip(filtered, 0.0, 1.0), depth_gamma)

    rgb = np.repeat(filtered[..., None], 3, axis=2)
    Image.fromarray(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)).save(dst)


def _save_inverse_depth_preprocessed(
    src: Path,
    dst: Path,
    *,
    flip_orientation: bool,
    percentile_low: float,
    percentile_high: float,
    eps: float,
    filter_type: str,
    gaussian_sigma_256: float,
    median_kernel_size: int,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space_256: float,
    depth_gamma: float,
) -> None:
    if not (0.0 <= percentile_low < percentile_high <= 100.0):
        raise ValueError(
            f"inverse-depth percentiles invalid: low={percentile_low}, "
            f"high={percentile_high} (require 0 <= low < high <= 100)"
        )

    arr = _to_grayscale_array(src)

    # arr==0 is the C3VD invalid-pixel sentinel; exclude from percentile stats.
    # arr==65535 is a valid clamp and is kept.
    valid_mask = arr > 0.0
    z = np.maximum(arr, eps)
    disparity = 1.0 / z

    if valid_mask.any():
        d_min = float(np.percentile(disparity[valid_mask], percentile_low))
        d_max = float(np.percentile(disparity[valid_mask], percentile_high))
    else:
        d_min, d_max = 0.0, 1.0
    if d_max <= d_min:
        d_max = d_min + 1e-8

    out = np.clip((disparity - d_min) / (d_max - d_min), 0.0, 1.0).astype(np.float32)
    # near=bright by construction (disparity = 1/z); flip to the far=bright
    # orientation when keep-orient mode is requested.
    if not flip_orientation:
        out = 1.0 - out

    filtered = _apply_filter_pipeline(
        out,
        filter_type=filter_type,
        gaussian_sigma_256=gaussian_sigma_256,
        median_kernel_size=median_kernel_size,
        bilateral_diameter=bilateral_diameter,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space_256=bilateral_sigma_space_256,
    )

    min_val = float(filtered.min())
    max_val = float(filtered.max())
    if max_val > min_val:
        filtered = (filtered - min_val) / (max_val - min_val)
    else:
        filtered = np.zeros_like(filtered, dtype=np.float32)

    if depth_gamma != 1.0:
        filtered = np.power(np.clip(filtered, 0.0, 1.0), depth_gamma)

    rgb = np.repeat(filtered[..., None], 3, axis=2)
    Image.fromarray(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)).save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=49, help="Must satisfy frames%%8==1")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument(
        "--domain",
        type=str,
        default="synthetic",
        help="Filter clips by domain field. Use 'any' to disable filter.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument(
        "--depth-suffix",
        type=str,
        default="_depth",
        help=(
            "Suffix inserted before the file extension when a depth_frames entry "
            "does not exist as-is."
        ),
    )
    parser.add_argument(
        "--caption-override-pool",
        type=str,
        nargs="+",
        default=None,
        help="If set, override each clip caption with a uniform draw from this pool.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        type=Path,
        default=None,
        help="If set, record cached_condition_path={prompt-cache-dir}/{slug(caption)}.pt per clip.",
    )
    parser.add_argument(
        "--depth-preprocess-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "c3vd_blur_renorm",
            "c3vd_inverse_depth_keep_orient",
            "c3vd_inverse_depth_flip_orient",
        ],
        help="How to stage reference depth frames under depth_frames/{clip_name}.",
    )
    parser.add_argument(
        "--gaussian-sigma-256",
        type=float,
        default=1.75,
        help=(
            "Gaussian sigma referenced to a 256x256 frame. The actual sigma scales "
            "with min(height, width)/256 for the source frame."
        ),
    )
    parser.add_argument(
        "--depth-filter-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "median_gaussian", "bilateral"],
        help=(
            "Filter pipeline used when --depth-preprocess-mode is enabled: "
            "'gaussian', 'median_gaussian', or 'bilateral'."
        ),
    )
    parser.add_argument(
        "--median-kernel-size",
        type=int,
        default=3,
        help="Median-filter kernel size for --depth-filter-type median_gaussian.",
    )
    parser.add_argument(
        "--bilateral-diameter",
        type=int,
        default=9,
        help="Pixel neighborhood diameter for --depth-filter-type bilateral.",
    )
    parser.add_argument(
        "--bilateral-sigma-color",
        type=float,
        default=0.1,
        help=(
            "Range sigma for bilateral filtering, operating on normalized depth in [0, 1]."
        ),
    )
    parser.add_argument(
        "--bilateral-sigma-space-256",
        type=float,
        default=3.0,
        help=(
            "Spatial sigma for bilateral filtering referenced to a 256x256 frame. "
            "The actual sigma scales with min(height, width)/256."
        ),
    )
    parser.add_argument(
        "--depth-gamma",
        type=float,
        default=1.0,
        help=(
            "Gamma compression/expansion applied after blur + renormalization. "
            "For example, 1.5 applies x -> x^1.5."
        ),
    )
    parser.add_argument(
        "--inverse-depth-percentile-low",
        type=float,
        default=2.0,
        help=(
            "Lower percentile (over valid pixels, per frame) used to set the "
            "low end of the disparity stretch in the c3vd_inverse_depth_* modes. "
            "Inert under other --depth-preprocess-mode values."
        ),
    )
    parser.add_argument(
        "--inverse-depth-percentile-high",
        type=float,
        default=98.0,
        help=(
            "Upper percentile (over valid pixels, per frame) used to set the "
            "high end of the disparity stretch in the c3vd_inverse_depth_* modes."
        ),
    )
    parser.add_argument(
        "--inverse-depth-eps",
        type=float,
        default=1.0,
        help=(
            "Floor (in raw depth units) applied via max(z, eps) before computing "
            "1/z, to avoid division blowups on very-near pixels in the "
            "c3vd_inverse_depth_* modes."
        ),
    )
    args = parser.parse_args()

    if args.num_frames % 8 != 1:
        sys.exit(f"ERROR: --num-frames must satisfy frames%%8==1, got {args.num_frames}")

    if not args.clips_json.exists():
        sys.exit(f"ERROR: clips JSON not found: {args.clips_json}")

    with open(args.clips_json) as f:
        all_clips = json.load(f)

    domain_filter = None if args.domain == "any" else args.domain
    pool = [c for c in all_clips if _eligible(c, args.num_frames, domain_filter)]
    print(f"Total clips in JSON: {len(all_clips)}; eligible after filter: {len(pool)}")
    if len(pool) < args.num_clips:
        sys.exit(f"ERROR: only {len(pool)} eligible clips, need {args.num_clips}")

    rng = random.Random(args.seed)
    selected = rng.sample(pool, args.num_clips)
    caption_rng = random.Random(args.seed + 1)

    first_frames_dir = args.output_dir / "first_frames"
    depth_root = args.output_dir / "depth_frames"
    source_rgb_root = args.output_dir / "source_rgb_frames"
    first_frames_dir.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)
    source_rgb_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = []
    for clip in selected:
        name = clip["name"]
        rgb_dir = Path(clip["rgb_dir"])
        depth_dir = Path(clip["depth_dir"])
        offset = args.frame_offset
        rgb_window = clip["rgb_frames"][offset : offset + args.num_frames]
        depth_window = clip["depth_frames"][offset : offset + args.num_frames]
        if len(rgb_window) < args.num_frames or len(depth_window) < args.num_frames:
            skipped.append({"name": name, "reason": "offset+num_frames exceeds clip length"})
            continue

        first_rgb_src = rgb_dir / rgb_window[0]
        if not first_rgb_src.exists():
            skipped.append({"name": name, "reason": f"missing first RGB frame {first_rgb_src}"})
            continue

        first_frame_path = first_frames_dir / f"{name}{first_rgb_src.suffix}"
        _relink(first_rgb_src, first_frame_path)

        clip_depth_dir = depth_root / name
        clip_depth_dir.mkdir(parents=True, exist_ok=True)
        _clear_dir(clip_depth_dir)
        missing = False
        for idx, depth_name in enumerate(depth_window):
            src = _resolve_depth_frame(depth_dir, depth_name, args.depth_suffix)
            if src is None:
                skipped.append(
                    {"name": name, "reason": f"missing depth frame {depth_dir / depth_name}"}
                )
                missing = True
                break

            if args.depth_preprocess_mode == "none":
                dst = clip_depth_dir / f"{idx:04d}{src.suffix}"
                _relink(src, dst)
            elif args.depth_preprocess_mode == "c3vd_blur_renorm":
                dst = clip_depth_dir / f"{idx:04d}.png"
                _save_preprocessed_depth(
                    src,
                    dst,
                    filter_type=args.depth_filter_type,
                    gaussian_sigma_256=args.gaussian_sigma_256,
                    median_kernel_size=args.median_kernel_size,
                    bilateral_diameter=args.bilateral_diameter,
                    bilateral_sigma_color=args.bilateral_sigma_color,
                    bilateral_sigma_space_256=args.bilateral_sigma_space_256,
                    depth_gamma=args.depth_gamma,
                )
            else:
                dst = clip_depth_dir / f"{idx:04d}.png"
                _save_inverse_depth_preprocessed(
                    src,
                    dst,
                    flip_orientation=(
                        args.depth_preprocess_mode == "c3vd_inverse_depth_flip_orient"
                    ),
                    percentile_low=args.inverse_depth_percentile_low,
                    percentile_high=args.inverse_depth_percentile_high,
                    eps=args.inverse_depth_eps,
                    filter_type=args.depth_filter_type,
                    gaussian_sigma_256=args.gaussian_sigma_256,
                    median_kernel_size=args.median_kernel_size,
                    bilateral_diameter=args.bilateral_diameter,
                    bilateral_sigma_color=args.bilateral_sigma_color,
                    bilateral_sigma_space_256=args.bilateral_sigma_space_256,
                    depth_gamma=args.depth_gamma,
                )

        if missing:
            continue

        clip_source_rgb_dir = source_rgb_root / name
        clip_source_rgb_dir.mkdir(parents=True, exist_ok=True)
        _clear_dir(clip_source_rgb_dir)
        rgb_missing = False
        for idx, rgb_name in enumerate(rgb_window):
            src = rgb_dir / rgb_name
            if not src.exists():
                skipped.append(
                    {"name": name, "reason": f"missing source RGB frame {src}"}
                )
                rgb_missing = True
                break
            dst = clip_source_rgb_dir / f"{idx:04d}{src.suffix}"
            _relink(src, dst)
        if rgb_missing:
            continue

        if args.caption_override_pool:
            caption = caption_rng.choice(args.caption_override_pool)
        else:
            caption = clip["caption"]

        entry: dict[str, Any] = {
            "clip_name": name,
            "first_frame_path": str(first_frame_path),
            "depth_frames_dir": str(clip_depth_dir),
            "source_rgb_frames_dir": str(clip_source_rgb_dir),
            "caption": caption,
            "fps": args.fps,
            "num_frames": args.num_frames,
            "source_rgb_window": rgb_window,
            "source_depth_window": depth_window,
            "domain": clip.get("domain"),
            "depth_preprocess_mode": args.depth_preprocess_mode,
        }
        if args.depth_preprocess_mode != "none":
            entry["depth_filter_type"] = args.depth_filter_type
            entry["gaussian_sigma_256"] = args.gaussian_sigma_256
            entry["median_kernel_size"] = args.median_kernel_size
            entry["bilateral_diameter"] = args.bilateral_diameter
            entry["bilateral_sigma_color"] = args.bilateral_sigma_color
            entry["bilateral_sigma_space_256"] = args.bilateral_sigma_space_256
            entry["depth_gamma"] = args.depth_gamma
        if args.depth_preprocess_mode in (
            "c3vd_inverse_depth_keep_orient",
            "c3vd_inverse_depth_flip_orient",
        ):
            entry["inverse_depth_percentile_low"] = args.inverse_depth_percentile_low
            entry["inverse_depth_percentile_high"] = args.inverse_depth_percentile_high
            entry["inverse_depth_eps"] = args.inverse_depth_eps
            entry["inverse_depth_flip_orientation"] = (
                args.depth_preprocess_mode == "c3vd_inverse_depth_flip_orient"
            )
        if args.caption_override_pool:
            entry["source_caption"] = clip["caption"]
            entry["caption_style"] = _caption_style(caption)
        if args.prompt_cache_dir is not None:
            entry["cached_condition_path"] = str(
                args.prompt_cache_dir / f"{_slug(caption)}.pt"
            )
        manifest.append(entry)
        print(f"  Prepared {name}: first={first_frame_path.name}, depth={clip_depth_dir}")

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "num_selected": len(manifest),
        "num_skipped": len(skipped),
        "num_frames": args.num_frames,
        "fps": args.fps,
        "domain": args.domain,
        "seed": args.seed,
        "clips_json": str(args.clips_json),
        "depth_suffix": args.depth_suffix,
        "depth_preprocess_mode": args.depth_preprocess_mode,
        "depth_filter_type": args.depth_filter_type,
        "gaussian_sigma_256": args.gaussian_sigma_256,
        "median_kernel_size": args.median_kernel_size,
        "bilateral_diameter": args.bilateral_diameter,
        "bilateral_sigma_color": args.bilateral_sigma_color,
        "bilateral_sigma_space_256": args.bilateral_sigma_space_256,
        "depth_gamma": args.depth_gamma,
        "skipped": skipped,
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nPrepared {len(manifest)} clips (skipped {len(skipped)}). "
        f"Manifest: {manifest_path}"
    )
    if skipped:
        print(f"Skipped clips logged to {summary_path}")


if __name__ == "__main__":
    main()

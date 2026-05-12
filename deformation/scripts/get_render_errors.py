import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import yaml

ORIGINAL_PATH = '/media/durrlab/6767b2b5-ec25-4139-9791-6ae49640cbe4/C3VDv2/aligned_final'

FPS = 30
COLORMAP_NAME = "JET"


MODALITIES = [
    ("depth", "depth", "depth", "*_depth.tiff"),
    ("diffuse", "diffuse", "diffuse", "*_diffuse.png"),
    ("normals", "normals", "normals", "*_normals.tiff"),
    ("occlusion", "occlusions", "occlusion", "*_occlusion.png"),
    ("optical_flow", "optical_flow", "optical_flow", "*_flow.tiff"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute render comparison outputs for one geometry.")
    parser.add_argument("--config", required=True, help="Path to YAML config file used by the render scripts")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def resolve_roots_from_config(config, config_path):
    missing = [k for k in ("geometry", "reference_dir", "c3vd_input_path") if config.get(k) is None]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(missing)}")

    geometry = str(config["geometry"])
    config_stem = Path(config_path).stem
    reference_dir = Path(str(config["reference_dir"]))
    c3vd_input_path = Path(str(config["c3vd_input_path"]))

    # reference_dir is expected to be relative to ORIGINAL_PATH, matching other scripts.
    if reference_dir.is_absolute():
        reference_root = reference_dir
    else:
        reference_root = Path(ORIGINAL_PATH) / reference_dir / "render"

    target_root = c3vd_input_path / config_stem / "render"
    output_root = c3vd_input_path / config_stem / "render_comparison"
    return geometry, reference_root, target_root, output_root


def collect_frames(folder, pattern):
    files = sorted(glob.glob(os.path.join(str(folder), pattern)))
    return {os.path.basename(path): path for path in files}


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def is_scalar_image(image):
    return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)


def normalize_to_uint8(image, min_value=None, max_value=None):
    if min_value is None:
        min_value = float(np.min(image))
    if max_value is None:
        max_value = float(np.max(image))

    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image.astype(np.float32) - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_display_frame(image, min_value=None, max_value=None):
    if is_scalar_image(image):
        display = image
        if display.ndim == 3:
            display = display[:, :, 0]
        display = normalize_to_uint8(display, min_value=min_value, max_value=max_value)
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    if image.dtype != np.uint8:
        display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        display = display.astype(np.uint8)
    else:
        display = image.copy()

    if display.ndim == 2:
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    if display.shape[2] == 1:
        return cv2.cvtColor(display[:, :, 0], cv2.COLOR_GRAY2BGR)
    if display.shape[2] == 4:
        return cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
    return display[:, :, :3]


def to_error_frame(error, colormap, max_error):
    if max_error <= 0:
        error_uint8 = np.zeros(error.shape, dtype=np.uint8)
    else:
        error_uint8 = normalize_to_uint8(error, 0.0, max_error)
    return cv2.applyColorMap(error_uint8, colormap)


def make_writer(output_path, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, FPS, size, isColor=True)


def process_modality(reference_root, target_root, output_root, modality_name, reference_subdir, target_subdir, pattern, config_path):
    reference_dir = reference_root / reference_subdir
    target_dir = target_root / target_subdir

    if not reference_dir.exists():
        print(f"Skipping {modality_name}: missing reference folder {reference_dir}")
        return
    if not target_dir.exists():
        print(f"Skipping {modality_name}: missing target folder {target_dir}")
        return

    reference_frames = collect_frames(reference_dir, pattern)
    target_frames = collect_frames(target_dir, pattern)
    common_names = sorted(set(reference_frames) & set(target_frames))

    if not common_names:
        print(f"Skipping {modality_name}: no matching files found")
        return

    first_reference = load_image(reference_frames[common_names[0]])
    first_target = load_image(target_frames[common_names[0]])

    if first_reference.shape != first_target.shape:
        raise RuntimeError(
            f"{modality_name}: reference and target shapes do not match: "
            f"{first_reference.shape} vs {first_target.shape}"
        )

    height, width = first_reference.shape[:2]
    modality_output_dir = output_root / modality_name
    modality_output_dir.mkdir(parents=True, exist_ok=True)

    # add folder name to end of videos to disambiguate when multiple folders are processed for the same geometry: ex c1_ascending_t2_v2_high
    name = config_path.split('/')[-1].split('.')[0]
    reference_writer = make_writer(modality_output_dir / f"reference_{name}.mp4", (width, height))
    target_writer = make_writer(modality_output_dir / f"target_{name}.mp4", (width, height))
    comparison_writer = make_writer(modality_output_dir / f"comparison_{name}.mp4", (width * 3, height))

    scalar_mode = is_scalar_image(first_reference) and is_scalar_image(first_target)
    global_min = float("inf")
    global_max = float("-inf")
    global_error_max = 0.0

    if scalar_mode:
        for name in common_names:
            reference = load_image(reference_frames[name])
            target = load_image(target_frames[name])

            if reference.shape != target.shape:
                raise RuntimeError(
                    f"{modality_name}: shape mismatch for frame {name}: {reference.shape} vs {target.shape}"
                )

            reference_scalar = reference[:, :, 0] if reference.ndim == 3 else reference
            target_scalar = target[:, :, 0] if target.ndim == 3 else target

            global_min = min(global_min, float(np.min(reference_scalar)), float(np.min(target_scalar)))
            global_max = max(global_max, float(np.max(reference_scalar)), float(np.max(target_scalar)))
            global_error_max = max(global_error_max, float(np.max(np.abs(reference_scalar - target_scalar))))

        error_maps_dir = modality_output_dir / "error_maps"
        error_maps_dir.mkdir(parents=True, exist_ok=True)
        error_writer = make_writer(modality_output_dir / "error_map.mp4", (width, height))
        colormap = getattr(cv2, f"COLORMAP_{COLORMAP_NAME}")
        print(
            f"{modality_name}: using fixed scalar range [{global_min:.6f}, {global_max:.6f}] and video-wide error max {global_error_max:.6f}"
        )
    else:
        error_maps_dir = None
        error_writer = None
        colormap = None

    total_abs_error = 0.0
    total_pixels = 0

    for index, name in enumerate(common_names):
        reference = load_image(reference_frames[name])
        target = load_image(target_frames[name])

        if reference.shape != target.shape:
            raise RuntimeError(
                f"{modality_name}: shape mismatch for frame {name}: {reference.shape} vs {target.shape}"
            )

        if scalar_mode:
            reference_scalar = reference[:, :, 0] if reference.ndim == 3 else reference
            target_scalar = target[:, :, 0] if target.ndim == 3 else target
            abs_error = np.abs(reference_scalar - target_scalar)
            total_abs_error += float(np.sum(abs_error))
            total_pixels += abs_error.size

            reference_preview = to_display_frame(reference_scalar, global_min, global_max)
            target_preview = to_display_frame(target_scalar, global_min, global_max)
            error_preview = to_error_frame(abs_error, colormap, global_error_max)
            comparison_frame = np.concatenate([reference_preview, target_preview, error_preview], axis=1)

            error_output_path = error_maps_dir / name.replace(".tiff", "_error.png").replace(".png", "_error.png")
            cv2.imwrite(str(error_output_path), error_preview)
            error_writer.write(error_preview)
        else:
            reference_preview = to_display_frame(reference)
            target_preview = to_display_frame(target)
            diff_preview = cv2.absdiff(reference_preview, target_preview)
            comparison_frame = np.concatenate([reference_preview, target_preview, diff_preview], axis=1)

        reference_writer.write(reference_preview)
        target_writer.write(target_preview)
        comparison_writer.write(comparison_frame)

        if scalar_mode:
            frame_mean = float(np.mean(abs_error))
            frame_max = float(np.max(abs_error))
            # print(
            #     f"[{modality_name}] [{index + 1:04d}/{len(common_names):04d}] {name}: "
            #     f"mean_abs_error={frame_mean:.6f}, max_abs_error={frame_max:.6f}"
            # )
        else:
            print(f"[{modality_name}] [{index + 1:04d}/{len(common_names):04d}] {name}")

    reference_writer.release()
    target_writer.release()
    comparison_writer.release()
    if error_writer is not None:
        error_writer.release()

    print(f"Saved {modality_name} reference video to {modality_output_dir / 'reference.mp4'}")
    print(f"Saved {modality_name} target video to {modality_output_dir / 'target.mp4'}")
    print(f"Saved {modality_name} comparison video to {modality_output_dir / 'comparison.mp4'}")

    if scalar_mode:
        overall_mean_abs_error = total_abs_error / total_pixels if total_pixels else 0.0
        print(f"Saved {modality_name} error video to {modality_output_dir / 'error_map.mp4'}")
        print(f"Saved {modality_name} per-frame error maps to {error_maps_dir}")
        print(f"{modality_name}: frames compared={len(common_names)}")
        print(f"{modality_name}: overall mean absolute error={overall_mean_abs_error:.6f}")
        print(f"{modality_name}: overall max absolute error={global_error_max:.6f}")


def main():
    args = parse_args()

    config = load_config(args.config)
    geometry, reference_root, target_root, output_root = resolve_roots_from_config(config, args.config)
    print(f"Using config {args.config} for geometry {geometry}")

    if not reference_root.exists():
        raise RuntimeError(f"Reference render root not found: {reference_root}")
    if not target_root.exists():
        raise RuntimeError(f"Target render root not found: {target_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for modality_name, reference_subdir, target_subdir, pattern in MODALITIES:
        process_modality(reference_root, target_root, output_root, modality_name, reference_subdir, target_subdir, pattern, args.config)


if __name__ == "__main__":
    main()
import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from helpers import (
    build_vertex_adjacency,
    compute_vertex_normals,
    interpolate_centerline_points,
    laplacian_smooth_scalar,
    preprocess_mesh_for_smoother_render,
    project_vertices_to_centerline_arclength,
    ras_to_lps,
    write_topology_obj,
)


CM_PER_UNIT = 0.1


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return config


def parse_waves(config):
    waves = config.get("waves")
    if waves is None:
        return [
            {
                "A": float(config.get("A", 0.4)),
                "sigma_frac": float(config.get("sigma_frac", 0.1)),
                "velocity_cm_s": float(config.get("velocity_cm_s", 1.0)),
                "start_delay_s": 0.0,
            }
        ]

    if not isinstance(waves, list) or len(waves) == 0:
        raise ValueError("'waves' must be a non-empty list when provided")

    default_A = float(config.get("A", 0.4))
    default_sigma_frac = float(config.get("sigma_frac", 0.1))
    default_velocity_cm_s = float(config.get("velocity_cm_s", 1.0))

    parsed = []
    for idx, wave in enumerate(waves):
        if not isinstance(wave, dict):
            raise ValueError(f"waves[{idx}] must be a YAML mapping")

        parsed.append(
            {
                "A": float(wave.get("A", default_A)),
                "sigma_frac": float(wave.get("sigma_frac", default_sigma_frac)),
                "velocity_cm_s": float(wave.get("velocity_cm_s", default_velocity_cm_s)),
                "start_delay_s": float(wave.get("start_delay_s", 0.0)),
            }
        )

    return parsed


def require_config_path(config, key):
    value = config.get(key)
    if value is None:
        raise RuntimeError(f"Missing required config field: {key}")
    return Path(str(value)).expanduser()


def resolve_reference_dir(config):
    reference_dir = require_config_path(config, "reference_dir")
    if not reference_dir.is_dir():
        raise RuntimeError(f"Missing reference directory: {reference_dir}")
    return reference_dir


def resolve_output_dir(config, config_stem):
    return require_config_path(config, "output_root") / config_stem


def resolve_centerline_path(config):
    centerline_path = config.get("centerline_path")
    if centerline_path is None:
        raise RuntimeError("Missing required config field: centerline_path")
    centerline_path = Path(str(centerline_path)).expanduser()
    if not centerline_path.exists():
        raise RuntimeError(f"Missing centerline: {centerline_path}")
    return centerline_path


def count_rgb_frames(reference_dir):
    rgb_dir = reference_dir / "rgb"

    if not rgb_dir.is_dir():
        raise RuntimeError(f"Missing rgb directory: {rgb_dir}")

    n_frames = sum(1 for path in rgb_dir.iterdir() if path.is_file())
    if n_frames == 0:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")

    return n_frames


def load_mesh(reference_dir, taubin_iterations=12, subdivision_iterations=0):
    mesh_path = reference_dir / "coverage_mesh.obj"
    if not mesh_path.exists():
        raise RuntimeError(f"Missing mesh: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh = preprocess_mesh_for_smoother_render(
        mesh,
        taubin_iterations=taubin_iterations,
        subdivision_iterations=subdivision_iterations,
    )

    return mesh_path, mesh


def build_projection_data(mesh, centerline_lps, smooth_vertex_s=True, vertex_s_smooth_iterations=12, vertex_s_smooth_lambda=0.45):
    diffs = np.diff(centerline_lps, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s_vals = np.concatenate([[0], np.cumsum(seg_lengths)])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_normals = np.asarray(mesh.vertex_normals)
    vertex_s = project_vertices_to_centerline_arclength(vertices, centerline_lps, s_vals)

    if smooth_vertex_s:
        neighbors = build_vertex_adjacency(len(vertices), triangles)
        vertex_s = laplacian_smooth_scalar(
            vertex_s,
            neighbors,
            iterations=vertex_s_smooth_iterations,
            lamb=vertex_s_smooth_lambda,
        )

    projected_points = interpolate_centerline_points(centerline_lps, s_vals, vertex_s)
    return {
        "vertices": vertices,
        "triangles": triangles,
        "vertex_normals": vertex_normals,
        "vertex_s": vertex_s,
        "projected_points": projected_points,
        "s_vals": s_vals,
        "mesh_s_min": float(vertex_s.min()),
        "mesh_s_max": float(vertex_s.max()),
        "s_total": float(s_vals[-1]),
    }


def build_runtime_waves(waves, s_total, cm_per_unit):
    waves_runtime = []
    for wave in waves:
        waves_runtime.append(
            {
                "A": float(wave["A"]),
                "sigma": max(float(wave["sigma_frac"]) * max(s_total, 1e-12), 1e-12),
                "velocity_units_s": float(wave["velocity_cm_s"]) / cm_per_unit,
                "start_delay_s": float(wave["start_delay_s"]),
            }
        )

    return waves_runtime


def print_runtime_summary(waves_runtime, fps, n_frames, s_total, mesh_s_min, mesh_s_max, cm_per_unit):
    for idx, wave in enumerate(waves_runtime):
        print(
            f"  Wave {idx + 1}: A={wave['A']:.3f}, sigma={wave['sigma']:.3f}, "
            f"velocity={wave['velocity_units_s'] * cm_per_unit:.3f} cm/s, "
            f"start_delay_s={wave['start_delay_s']:.3f}"
        )

    print("Physical speed: per-wave (see wave summary above)")
    print(f"Centerline length: {s_total * cm_per_unit:.3f} cm")
    print(
        f"Mesh centerline coverage: {mesh_s_min * cm_per_unit:.3f} cm to "
        f"{mesh_s_max * cm_per_unit:.3f} cm"
    )
    print(f"Export frames: {n_frames} at {fps} fps")


def generate_frames(
    projection_data,
    waves_runtime,
    n_frames,
    fps,
    debug_vertex_bounds,
):
    vertices = projection_data["vertices"]
    triangles = projection_data["triangles"]
    vertex_s = projection_data["vertex_s"]
    projected_points = projection_data["projected_points"]
    s_total = projection_data["s_total"]

    all_frame_vertices = np.empty((n_frames, len(vertices), 3), dtype=np.float32)
    all_frame_normals = np.empty((n_frames, len(vertices), 3), dtype=np.float32)

    time_start = time.time()
    for frame in range(n_frames):
        time_elapsed = time.time() - time_start
        print(f"Processing frame {frame + 1}/{n_frames}... (Elapsed time: {time_elapsed:.2f} s)")
        t = frame / fps
        contraction = np.zeros_like(vertex_s, dtype=np.float64)
        for wave in waves_runtime:
            if t < wave["start_delay_s"]:
                continue

            t_wave = t - wave["start_delay_s"]
            s_center = (wave["velocity_units_s"] * t_wave) % s_total
            gauss = np.exp(-0.5 * ((vertex_s - s_center) / wave["sigma"]) ** 2)
            contraction += wave["A"] * gauss

        contraction = np.clip(contraction, 0.0, 0.95)
        radius_factor = 1.0 - contraction
        frame_vertices = projected_points + (vertices - projected_points) * radius_factor[:, None]

        if debug_vertex_bounds and (frame < 3 or frame == n_frames - 1 or frame % 30 == 0):
            frame_min = frame_vertices.min(axis=0)
            frame_max = frame_vertices.max(axis=0)
            frame_range = frame_max - frame_min
            print(
                f"  Bounds frame {frame + 1}: min={frame_min.tolist()}, max={frame_max.tolist()}, range={frame_range.tolist()}"
            )

        all_frame_vertices[frame] = frame_vertices.astype(np.float32)
        all_frame_normals[frame] = compute_vertex_normals(frame_vertices, triangles)

    return all_frame_vertices, all_frame_normals


def export_geometry(output_dir, projection_data, all_frame_vertices, all_frame_normals):
    vertices = projection_data["vertices"]
    triangles = projection_data["triangles"]
    vertex_normals = projection_data["vertex_normals"]

    output_dir.mkdir(parents=True, exist_ok=True)

    connectivity_obj_path = output_dir / "model.obj"
    vertex_bin_path = output_dir / "vertex_positions.bin"
    normals_bin_path = output_dir / "vertex_normals.bin"

    write_topology_obj(str(connectivity_obj_path), vertices, triangles, vertex_normals)
    all_frame_vertices.tofile(str(vertex_bin_path))
    all_frame_normals.tofile(str(normals_bin_path))

    print("Saved smoothed gaussian export:")
    print(f"  Topology OBJ: {connectivity_obj_path}")
    print(f"  Vertex binary: {vertex_bin_path}")
    print(f"  Vertex normals binary: {normals_bin_path}")
    print(f"  Shape (frames, vertices, xyz): {all_frame_vertices.shape}")
    print(f"  Length of centerline (units): {projection_data['s_total']:.3f}")
    print(
        f"  Mesh centerline coverage (units): "
        f"{projection_data['mesh_s_min']:.3f} to {projection_data['mesh_s_max']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Export smoothed C3VD Gaussian animations.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    config_stem = Path(args.config).stem

    geometry = str(config["geometry"])
    waves = parse_waves(config)
    fps = float(config.get("fps", 29.97))
    debug_vertex_bounds = bool(config.get("debug_vertex_bounds", False))
    taubin_iterations = int(config.get("taubin_iterations", 12))
    subdivision_iterations = int(config.get("subdivision_iterations", 0))
    smooth_vertex_s = bool(config.get("smooth_vertex_s", True))
    vertex_s_smooth_iterations = int(config.get("vertex_s_smooth_iterations", 12))
    vertex_s_smooth_lambda = float(config.get("vertex_s_smooth_lambda", 0.45))

    reference_dir = resolve_reference_dir(config)
    output_dir = resolve_output_dir(config, config_stem)
    n_frames = count_rgb_frames(reference_dir)
    mesh_path, mesh = load_mesh(
        reference_dir,
        taubin_iterations=taubin_iterations,
        subdivision_iterations=subdivision_iterations,
    )
    centerline_path = resolve_centerline_path(config)
    centerline = np.load(centerline_path)
    centerline_lps = ras_to_lps(centerline)
    centerline = np.load(centerline_path)
    centerline_lps = ras_to_lps(centerline)

    print(f"Processing geometry: {geometry}, waves={len(waves)}, fps={fps}")
    print(f"Reference directory: {reference_dir}")
    print(f"Mesh path: {mesh_path}")
    print(f"Centerline path: {centerline_path}")

    projection_data = build_projection_data(
        mesh,
        centerline_lps,
        smooth_vertex_s=smooth_vertex_s,
        vertex_s_smooth_iterations=vertex_s_smooth_iterations,
        vertex_s_smooth_lambda=vertex_s_smooth_lambda,
    )
    waves_runtime = build_runtime_waves(waves, projection_data["s_total"], CM_PER_UNIT)
    print_runtime_summary(
        waves_runtime,
        fps,
        n_frames,
        projection_data["s_total"],
        projection_data["mesh_s_min"],
        projection_data["mesh_s_max"],
        CM_PER_UNIT,
    )

    all_frame_vertices, all_frame_normals = generate_frames(
        projection_data,
        waves_runtime,
        n_frames,
        fps,
        debug_vertex_bounds,
    )

    export_geometry(output_dir, projection_data, all_frame_vertices, all_frame_normals)


if __name__ == "__main__":
    print("Starting C3VD peristalsis generation...")
    main()
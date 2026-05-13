import argparse
import os
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
    lps_to_ras,
    preprocess_mesh_for_smoother_render,
    project_vertices_to_centerline_arclength,
    ras_to_lps,
    resample_even_arc,
    write_topology_obj,
)


# ---------- FIXED PARAMS ----------
N_SAMPLES = 1000
RESAMPLE_FACTOR = 1
SMOOTH_WINDOW_DIVISOR = 10

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def smooth_centerline(points, window=9):
    if window <= 1:
        return points.copy()
    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(points)
    for i in range(3):
        out[:, i] = np.convolve(padded[:, i], kernel, mode="valid")

    out[0] = points[0]
    out[-1] = points[-1]
    return out


def compute_optimal_deformation_axis(centerline_lps):
    """Return the dominant transverse PCA direction for a centerline in LPS space."""
    c = resample_even_arc(
        np.asarray(centerline_lps, dtype=float),
        n_out=max(3, min(1000, len(centerline_lps) * 4)),
    )
    centered = c - c.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    if len(singular_values) < 2 or singular_values[1] <= 1e-12:
        tangents = compute_centerline_tangents(c)
        tangent = tangents.mean(axis=0)
        tangent /= np.linalg.norm(tangent) + 1e-12
        axes = np.eye(3)
        axis = axes[np.argmin(np.abs(axes @ tangent))]
    else:
        axis = vh[1]

    axis = axis / (np.linalg.norm(axis) + 1e-12)
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    return axis


def compute_centerline_tangents(points):
    tangents = np.zeros_like(points, dtype=float)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    return tangents


def rotate_vector_about_axis(v, axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
    )


def parallel_transport_normals(tangents, preferred_axis):
    n_pts = len(tangents)
    normals = np.zeros_like(tangents)

    preferred = preferred_axis / (np.linalg.norm(preferred_axis) + 1e-12)
    t0 = tangents[0]

    n0 = preferred - np.dot(preferred, t0) * t0
    if np.linalg.norm(n0) < 1e-8:
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dots = np.abs(axes @ t0)
        alt = axes[np.argmin(dots)]
        n0 = alt - np.dot(alt, t0) * t0
    normals[0] = n0 / (np.linalg.norm(n0) + 1e-12)

    for i in range(1, n_pts):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        axis = np.cross(t_prev, t_curr)
        axis_norm = np.linalg.norm(axis)

        n_prev = normals[i - 1]
        if axis_norm < 1e-10:
            n_curr = n_prev
        else:
            dot_tc = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            angle = np.arccos(dot_tc)
            n_curr = rotate_vector_about_axis(n_prev, axis / axis_norm, angle)

        n_curr = n_curr - np.dot(n_curr, t_curr) * t_curr
        normals[i] = n_curr / (np.linalg.norm(n_curr) + 1e-12)

    return normals


def warp_centerline_half_wave(
    centerline, amplitude=1.0, phase=0.0, axis=np.array([0.0, 0.0, 1.0])
):
    c = centerline.astype(float).copy()
    diffs = np.diff(c, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = s[-1]
    if total_len <= 1e-12:
        return c

    disp = amplitude * np.sin(np.pi * s / total_len + phase)
    tangents = compute_centerline_tangents(c)
    normals = parallel_transport_normals(tangents, axis)
    return c + disp[:, None] * normals


def warp_centerline_exp_tail(
    centerline,
    amplitude=4.0,
    axis=np.array([0.0, 1.0, 0.0]),
    sharpness=4.0,
    start_fraction=0.5,
):
    c = centerline.astype(float).copy()
    seg_len = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = s[-1]
    if total_len <= 1e-12:
        return c

    u = s / total_len
    sf = float(np.clip(start_fraction, 0.0, 0.999999))
    tail = np.clip((u - sf) / (1.0 - sf), 0.0, 1.0)

    k = float(sharpness)
    if abs(k) < 1e-8:
        exp_ramp = tail
    else:
        exp_ramp = (np.exp(k * tail) - 1.0) / (np.exp(k) - 1.0)

    disp = amplitude * exp_ramp
    tangents = compute_centerline_tangents(c)
    normals = parallel_transport_normals(tangents, axis)
    return c + disp[:, None] * normals


def shift_centerline_constant(
    centerline, direction=np.array([0.0, 1.0, 0.0]), distance=3.0
):
    d = np.asarray(direction, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm <= 1e-12:
        raise ValueError("SHIFT_DIRECTION must be non-zero")
    offset = (distance / d_norm) * d
    return centerline.astype(float).copy() + offset[None, :]


def shift_centerline_linear(
    centerline,
    direction=np.array([0.0, 1.0, 0.0]),
    distance=3.0,
    fixed_end="start",
):
    d = np.asarray(direction, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm <= 1e-12:
        raise ValueError("SHIFT_DIRECTION must be non-zero")

    c = centerline.astype(float).copy()
    seg_len = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = s[-1]

    if total_len <= 1e-12:
        ramp = np.zeros(len(c), dtype=float)
    else:
        ramp = s / total_len

    if fixed_end == "end":
        ramp = 1.0 - ramp
    elif fixed_end != "start":
        raise ValueError("SHIFT_LINEAR_FIXED_END must be 'start' or 'end'")

    unit_d = d / d_norm
    return c + (distance * ramp)[:, None] * unit_d[None, :]


def generate_warped_centerline(centerline_lps, transform_mode, transform_params):
    centerline_resampled = resample_even_arc(
        centerline_lps,
        n_out=max(2, len(centerline_lps) * RESAMPLE_FACTOR),
    )

    smooth_window = max(
        1, len(centerline_resampled) // max(int(SMOOTH_WINDOW_DIVISOR), 1)
    )
    centerline_smooth = smooth_centerline(centerline_resampled, window=smooth_window)
    auto_axis = compute_optimal_deformation_axis(centerline_smooth)

    direction_param = transform_params.get("direction", [0.0, 1.0, 0.0])
    if isinstance(direction_param, str) and direction_param.lower() == "auto":
        direction_param = auto_axis.tolist()

    axis_param = transform_params.get("axis", [0.0, 1.0, 0.0])
    if isinstance(axis_param, str) and axis_param.lower() == "auto":
        axis_param = auto_axis.tolist()

    if transform_mode == "shift":
        centerline_warped = shift_centerline_constant(
            centerline_smooth,
            direction=np.asarray(direction_param, dtype=float),
            distance=float(transform_params.get("distance", 5.0)),
        )
    elif transform_mode == "linear_shift":
        centerline_warped = shift_centerline_linear(
            centerline_smooth,
            direction=np.asarray(direction_param, dtype=float),
            distance=float(transform_params.get("distance", 5.0)),
            fixed_end=str(transform_params.get("fixed_end", "start")),
        )
    elif transform_mode == "warp":
        centerline_warped = warp_centerline_half_wave(
            centerline_smooth,
            amplitude=float(transform_params.get("amplitude", 8.0)),
            phase=float(transform_params.get("phase", 0.0)),
            axis=np.asarray(axis_param, dtype=float),
        )
    elif transform_mode == "exp_tail_warp":
        centerline_warped = warp_centerline_exp_tail(
            centerline_smooth,
            amplitude=float(transform_params.get("amplitude", 8.0)),
            axis=np.asarray(axis_param, dtype=float),
            sharpness=float(transform_params.get("sharpness", 5.0)),
            start_fraction=float(transform_params.get("start_fraction", 0.05)),
        )
    else:
        raise ValueError(
            "transform_mode must be 'shift', 'linear_shift', 'warp', or 'exp_tail_warp'"
        )

    return centerline_smooth, centerline_warped


def encode_global_offsets(vertices, src_cl, triangles, smooth_vertex_s=True, vertex_s_smooth_iterations=24, vertex_s_smooth_lambda=0.55):
    diffs = np.diff(src_cl, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s_vals = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    s_total = max(float(s_vals[-1]), 1e-12)

    vertex_s = project_vertices_to_centerline_arclength(vertices, src_cl, s_vals)
    if smooth_vertex_s:
        neighbors = build_vertex_adjacency(len(vertices), triangles)
        vertex_s = laplacian_smooth_scalar(
            vertex_s,
            neighbors,
            iterations=vertex_s_smooth_iterations,
            lamb=vertex_s_smooth_lambda,
        )

    s_frac = np.clip(vertex_s / s_total, 0.0, 1.0)
    src_anchor = interpolate_centerline_points(src_cl, s_vals, s_frac * s_total)

    delta_xyz = vertices - src_anchor
    return s_frac, delta_xyz


def decode_global_offsets(s_frac, delta_xyz, dst_cl):
    diffs = np.diff(dst_cl, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s_vals = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    s_total = max(float(s_vals[-1]), 1e-12)

    dst_anchor = interpolate_centerline_points(
        dst_cl, s_vals, np.clip(s_frac, 0.0, 1.0) * s_total
    )
    return dst_anchor + delta_xyz


def ease_in_out(alpha):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0))


def resolve_reference_dir(config):
    reference = str(config["reference_dir"])
    reference_path = Path(str(reference)).expanduser()

    if reference_path.is_absolute():
        if not reference_path.is_dir():
            raise RuntimeError(f"Missing reference directory: {reference_path}")
        return reference_path

    reference_base = Path(str(config.get("reference_base"))).expanduser()
    reference_path = reference_base / reference
    if not reference_path.is_dir():
        raise RuntimeError(f"Missing reference directory: {reference_path}")
    return reference_path


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


def process_geometry(
    geometry,
    mesh,
    cl_src_smooth,
    cl_dst_warped,
    n_rgb_frames,
    output_dir,
    fps,
    debug_vertex_bounds=False,
    smooth_vertex_s=True,
    vertex_s_smooth_iterations=24,
    vertex_s_smooth_lambda=0.55,
):
    vertices_src = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    cl_src = resample_even_arc(cl_src_smooth, N_SAMPLES)
    cl_dst = resample_even_arc(cl_dst_warped, N_SAMPLES)

    s_frac, delta_xyz = encode_global_offsets(
        vertices_src,
        cl_src,
        triangles,
        smooth_vertex_s=smooth_vertex_s,
        vertex_s_smooth_iterations=vertex_s_smooth_iterations,
        vertex_s_smooth_lambda=vertex_s_smooth_lambda,
    )

    n_frames = max(2, int(n_rgb_frames))

    all_frame_vertices = np.empty((n_frames, len(vertices_src), 3), dtype=np.float32)
    all_frame_normals = np.empty((n_frames, len(vertices_src), 3), dtype=np.float32)

    print(f"Export frames for {geometry}: {n_frames} at {fps} fps")
    time_start = time.time()
    for frame in range(n_frames):
        time_elapsed = time.time() - time_start
        print(
            f"  Processing frame {frame + 1}/{n_frames} (Elapsed time: {time_elapsed:.2f} s)",
            end="\r",
        )
        phase_frame = frame / max(float(n_frames - 1), 1.0)
        alpha_frame = ease_in_out(phase_frame)

        cl_frame = (1.0 - alpha_frame) * cl_src + alpha_frame * cl_dst
        frame_vertices = decode_global_offsets(s_frac, delta_xyz, cl_frame)

        if debug_vertex_bounds and (
            frame < 3 or frame == n_frames - 1 or frame % 30 == 0
        ):
            frame_min = frame_vertices.min(axis=0)
            frame_max = frame_vertices.max(axis=0)
            frame_range = frame_max - frame_min
            print(
                f"  Bounds frame {frame + 1}: min={frame_min.tolist()}, max={frame_max.tolist()}, range={frame_range.tolist()}"
            )

        all_frame_vertices[frame] = frame_vertices.astype(np.float32)
        all_frame_normals[frame] = compute_vertex_normals(frame_vertices, triangles)

    output_dir.mkdir(parents=True, exist_ok=True)

    connectivity_obj_path = output_dir / "model.obj"
    vertex_bin_path = output_dir / "vertex_positions.bin"
    normals_bin_path = output_dir / "vertex_normals.bin"

    write_topology_obj(
        str(connectivity_obj_path),
        vertices_src,
        triangles,
        np.asarray(mesh.vertex_normals),
    )
    all_frame_vertices.tofile(str(vertex_bin_path))
    all_frame_normals.tofile(str(normals_bin_path))

    print("\nSaved centerline warp export:")
    print(f"  Topology OBJ: {connectivity_obj_path}")
    print(f"  Vertex binary: {vertex_bin_path}")
    print(f"  Vertex normals binary: {normals_bin_path}")
    print(f"  Shape (frames, vertices, xyz): {all_frame_vertices.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export C3VD centerline warp and smooth geometry."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    config_stem = Path(args.config).stem

    geometry = str(config["geometry"])
    transform_mode = str(config["transform_mode"])
    transform_params = config.get("transform_params", {})
    fps = int(config.get("fps", 30))
    save_new_centerline = bool(config.get("save_new_centerline", True))
    debug_vertex_bounds = bool(config.get("debug_vertex_bounds", False))

    if transform_mode not in {"shift", "linear_shift", "warp", "exp_tail_warp"}:
        raise RuntimeError(
            "transform_mode must be one of: shift, linear_shift, warp, exp_tail_warp"
        )
    if not isinstance(transform_params, dict):
        raise RuntimeError("transform_params must be a dictionary")

    taubin_iterations = int(config.get("taubin_iterations", 12))
    subdivision_iterations = int(config.get("subdivision_iterations", 0))
    smooth_vertex_s = bool(config.get("smooth_vertex_s", True))
    vertex_s_smooth_iterations = int(config.get("vertex_s_smooth_iterations", 24))
    vertex_s_smooth_lambda = float(config.get("vertex_s_smooth_lambda", 0.55))

    reference_dir = resolve_reference_dir(config)
    centerline_path = resolve_centerline_path(config)
    n_rgb_frames = count_rgb_frames(reference_dir)
    mesh_path, mesh = load_mesh(
        reference_dir,
        taubin_iterations=taubin_iterations,
        subdivision_iterations=subdivision_iterations,
    )

    output_dir = (
        Path(str(config.get("output_root"))) / config_stem
    )

    print(f"\n=== Processing geometry: {geometry} ===")
    print(f"Reference directory: {reference_dir}")
    print(f"Centerline path: {centerline_path}")
    print(f"Mesh path: {mesh_path}")
    print(f"Found {n_rgb_frames} RGB frames")

    cl_src = np.load(centerline_path).astype(float)
    cl_src_lps = ras_to_lps(cl_src)

    cl_smooth_lps, cl_warped_lps = generate_warped_centerline(
        cl_src_lps, transform_mode, transform_params
    )

    warped_name = f"centerline_warped_{transform_mode}.npy"
    warped_centerline_path = output_dir / warped_name
    cl_to_save = lps_to_ras(cl_warped_lps)
    if save_new_centerline:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(warped_centerline_path), cl_to_save)
        print(f"Saved warped centerline: {warped_centerline_path} shape={cl_to_save.shape}")

    process_geometry(
        geometry=geometry,
        mesh=mesh,
        cl_src_smooth=cl_smooth_lps,
        cl_dst_warped=cl_warped_lps,
        n_rgb_frames=n_rgb_frames,
        output_dir=output_dir,
        fps=fps,
        debug_vertex_bounds=debug_vertex_bounds,
        smooth_vertex_s=smooth_vertex_s,
        vertex_s_smooth_iterations=vertex_s_smooth_iterations,
        vertex_s_smooth_lambda=vertex_s_smooth_lambda,
    )


if __name__ == "__main__":
    main()

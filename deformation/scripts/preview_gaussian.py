import argparse
import copy
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from deformation.scripts.generate_gaussian import (
    CM_PER_UNIT,
    build_runtime_waves,
    load_centerline,
    load_config,
    parse_waves,
    resolve_output_dir,
)


def load_export(output_dir):
    mesh_path = output_dir / "model.obj"
    vertex_bin_path = output_dir / "vertex_positions.bin"

    if not mesh_path.exists():
        raise RuntimeError(f"Missing exported mesh: {mesh_path}")
    if not vertex_bin_path.exists():
        raise RuntimeError(f"Missing exported vertex positions: {vertex_bin_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    n_vertices = len(mesh.vertices)
    if n_vertices == 0:
        raise RuntimeError(f"Exported mesh has no vertices: {mesh_path}")

    raw_vertices = np.fromfile(str(vertex_bin_path), dtype=np.float32)
    values_per_frame = n_vertices * 3
    if len(raw_vertices) == 0 or len(raw_vertices) % values_per_frame != 0:
        raise RuntimeError(
            f"Vertex binary size does not match mesh vertex count: {vertex_bin_path}"
        )

    frame_vertices = raw_vertices.reshape((-1, n_vertices, 3))
    return mesh, frame_vertices


def centerline_arclength(centerline_lps):
    diffs = np.diff(centerline_lps, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def build_centerline_geometry(centerline_lps):
    points = o3d.utility.Vector3dVector(centerline_lps)
    lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(centerline_lps.shape[0] - 1)])
    centerline_set = o3d.geometry.LineSet(points=points, lines=lines)
    centerline_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * (centerline_lps.shape[0] - 1))

    spheres = []
    for point in [centerline_lps[0], centerline_lps[-1]]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.translate(point)
        spheres.append(sphere)

    return centerline_set, spheres


def first_visible_wave_center(waves_runtime, t, s_total, boundary_fade_sigma, time_fade_s):
    for wave in waves_runtime:
        if t < wave["start_delay_s"]:
            continue

        boundary_margin = boundary_fade_sigma * wave["sigma"]
        t_wave = t - wave["start_delay_s"]
        travel = wave["velocity_units_s"] * t_wave
        period = s_total

        first_copy = int(np.floor((travel - s_total - boundary_margin) / period))
        last_copy = int(np.ceil((travel + boundary_margin) / period))

        for copy_idx in range(first_copy, last_copy + 1):
            s_center = travel - copy_idx * period
            if time_fade_s > 0:
                time_fade = np.clip(t_wave / max(time_fade_s, 1e-12), 0.0, 1.0)
                time_fade = time_fade * time_fade * (3.0 - 2.0 * time_fade)
            else:
                time_fade = 1.0

            if boundary_margin <= 0:
                boundary_fade = 1.0 if 0.0 <= s_center <= s_total else 0.0
            else:
                enter = np.clip((s_center + boundary_margin) / boundary_margin, 0.0, 1.0)
                enter = enter * enter * (3.0 - 2.0 * enter)
                exit_ = np.clip((s_center - s_total) / boundary_margin, 0.0, 1.0)
                exit_ = 1.0 - (exit_ * exit_ * (3.0 - 2.0 * exit_))
                boundary_fade = float(enter * exit_)

            if boundary_fade * time_fade > 1e-6:
                return s_center

    return 0.0


def get_arrow_and_plane_time(centerline_lps, s_vals, s_center, bbox_diagonal):
    idx = np.searchsorted(s_vals, s_center)
    idx = np.clip(idx, 1, len(centerline_lps) - 1)
    start = centerline_lps[idx - 1]
    end = centerline_lps[idx]
    tangent = end - start
    tangent_norm = np.linalg.norm(tangent)
    direction = tangent / (tangent_norm if tangent_norm > 0 else 1)
    f = (s_center - s_vals[idx - 1]) / (s_vals[idx] - s_vals[idx - 1] + 1e-12)
    position = (1 - f) * start + f * end

    arrow_length = bbox_diagonal * 0.03
    arrow_radius = bbox_diagonal * 0.005
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=arrow_radius,
        cone_radius=arrow_radius * 2,
        cylinder_height=arrow_length * 0.6,
        cone_height=arrow_length * 0.4,
    )
    arrow.paint_uniform_color([0, 1, 0])

    plane_size = bbox_diagonal * 0.1
    thickness = bbox_diagonal * 0.002
    plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=thickness)
    plane.paint_uniform_color([0, 0, 1])
    plane_center_offset = np.array([plane_size / 2, plane_size / 2, thickness / 2])
    plane.translate(-plane_center_offset)

    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-6:
        rot_matrix = np.eye(3)
    else:
        axis /= np.linalg.norm(axis)
        rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    arrow.rotate(rot_matrix, center=(0, 0, 0))
    arrow.translate(position)
    plane.rotate(rot_matrix, center=(0, 0, 0))
    plane.translate(position)
    return arrow, plane


def preview_animation(
    mesh,
    frame_vertices,
    centerline_lps,
    waves_runtime,
    fps,
    wave_boundary_fade_sigma,
    wave_time_fade_s,
):
    s_vals = centerline_arclength(centerline_lps)
    s_total = float(s_vals[-1])
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    centerline_set, spheres = build_centerline_geometry(centerline_lps)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="C3VD Gaussian Animation Preview (Press H to toggle original mesh)")

    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(mesh)
    vis.add_geometry(centerline_set)
    for sphere in spheres:
        vis.add_geometry(sphere)

    deformed_mesh = copy.deepcopy(mesh)
    deformed_mesh.paint_uniform_color([0.6, 0.4, 1])
    vis.add_geometry(deformed_mesh, reset_bounding_box=False)

    arrow, plane = get_arrow_and_plane_time(centerline_lps, s_vals, 0.0, bbox_diagonal)
    vis.add_geometry(arrow, reset_bounding_box=False)
    vis.add_geometry(plane, reset_bounding_box=False)

    show_original = [True]

    def toggle_original_mesh(vis_obj):
        if show_original[0]:
            vis_obj.remove_geometry(mesh, reset_bounding_box=False)
        else:
            vis_obj.add_geometry(mesh, reset_bounding_box=False)
        show_original[0] = not show_original[0]
        return False

    vis.register_key_callback(72, toggle_original_mesh)

    dt = 1.0 / fps
    frame_idx = 0
    n_frames = len(frame_vertices)
    while True:
        t = frame_idx / fps
        s_center = first_visible_wave_center(
            waves_runtime,
            t,
            s_total,
            wave_boundary_fade_sigma,
            wave_time_fade_s,
        )

        deformed_mesh.vertices = o3d.utility.Vector3dVector(frame_vertices[frame_idx].astype(np.float64))
        deformed_mesh.compute_vertex_normals()

        vis.remove_geometry(arrow, reset_bounding_box=False)
        vis.remove_geometry(plane, reset_bounding_box=False)
        arrow, plane = get_arrow_and_plane_time(centerline_lps, s_vals, s_center, bbox_diagonal)
        vis.add_geometry(arrow, reset_bounding_box=False)
        vis.add_geometry(plane, reset_bounding_box=False)

        vis.update_geometry(deformed_mesh)
        vis.update_geometry(centerline_set)
        for sphere in spheres:
            vis.update_geometry(sphere)
        vis.update_geometry(arrow)
        vis.update_geometry(plane)

        if not vis.poll_events():
            break
        vis.update_renderer()

        frame_idx = (frame_idx + 1) % n_frames
        time.sleep(dt)

    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Preview a generated C3VD Gaussian animation export.")
    parser.add_argument("--config", required=True, help="Path to the same YAML config used for generation")
    args = parser.parse_args()

    config = load_config(args.config)
    config_stem = Path(args.config).stem
    fps = float(config.get("fps", 29.97))
    wave_time_fade_s = float(config.get("wave_time_fade_s", 0.5))
    wave_boundary_fade_sigma = float(config.get("wave_boundary_fade_sigma", 3.0))

    output_dir = resolve_output_dir(config, config_stem)
    mesh, frame_vertices = load_export(output_dir)
    reference_dir = Path(str(config["reference_dir"])).expanduser()
    _, centerline_lps = load_centerline(config, reference_dir)
    s_total = float(centerline_arclength(centerline_lps)[-1])
    waves_runtime = build_runtime_waves(parse_waves(config), s_total, CM_PER_UNIT)

    print(f"Previewing export: {output_dir}")
    print(f"Frames: {len(frame_vertices)}, fps={fps}")
    preview_animation(
        mesh,
        frame_vertices,
        centerline_lps,
        waves_runtime,
        fps,
        wave_boundary_fade_sigma,
        wave_time_fade_s,
    )


if __name__ == "__main__":
    main()
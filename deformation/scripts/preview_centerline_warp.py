import argparse
import copy
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from deformation.scripts.generate_centerline_warp import (
    load_config,
    resolve_centerline_path,
    resolve_reference_dir,
    ras_to_lps,
    compute_centerline_tangents,
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


def build_centerline_geometry(centerline_lps):
    points = o3d.utility.Vector3dVector(centerline_lps)
    lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(centerline_lps.shape[0] - 1)]
    )
    centerline_set = o3d.geometry.LineSet(points=points, lines=lines)
    centerline_set.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0]] * (centerline_lps.shape[0] - 1)
    )

    spheres = []
    for point in [centerline_lps[0], centerline_lps[-1]]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.translate(point)
        spheres.append(sphere)

    return centerline_set, spheres


def get_arrow_and_plane_at_progress(centerline_lps, progress, bbox_diagonal):
    """
    Place an arrow and plane at a position along the centerline based on progress (0 to 1).
    """
    n_pts = len(centerline_lps)
    idx = int(np.clip(progress * (n_pts - 1), 0, n_pts - 1))
    idx = np.clip(idx, 0, n_pts - 1)
    position = centerline_lps[idx]

    if idx > 0:
        direction = centerline_lps[idx] - centerline_lps[idx - 1]
    elif idx < n_pts - 1:
        direction = centerline_lps[idx + 1] - centerline_lps[idx]
    else:
        direction = np.array([0.0, 0.0, 1.0])

    tangent_norm = np.linalg.norm(direction)
    direction = direction / (tangent_norm if tangent_norm > 0 else 1)

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
    plane = o3d.geometry.TriangleMesh.create_box(
        width=plane_size, height=plane_size, depth=thickness
    )
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
    centerline_src_lps,
    centerline_dst_lps,
    fps,
):
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_diagonal = np.linalg.norm(
        bbox.get_max_bound() - bbox.get_min_bound()
    )
    centerline_src_set, spheres_src = build_centerline_geometry(centerline_src_lps)
    centerline_dst_set, spheres_dst = build_centerline_geometry(centerline_dst_lps)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="C3VD Centerline Warp Preview (Press H to toggle original mesh)"
    )

    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(mesh)
    vis.add_geometry(centerline_src_set)
    for sphere in spheres_src:
        vis.add_geometry(sphere)

    deformed_mesh = copy.deepcopy(mesh)
    deformed_mesh.paint_uniform_color([0.6, 0.4, 1])
    vis.add_geometry(deformed_mesh, reset_bounding_box=False)

    centerline_dst_set.paint_uniform_color([0, 1, 0])
    vis.add_geometry(centerline_dst_set, reset_bounding_box=False)
    for sphere in spheres_dst:
        vis.add_geometry(sphere, reset_bounding_box=False)

    arrow, plane = get_arrow_and_plane_at_progress(
        centerline_src_lps, 0.0, bbox_diagonal
    )
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
        progress = frame_idx / max(n_frames - 1, 1.0)

        deformed_mesh.vertices = o3d.utility.Vector3dVector(
            frame_vertices[frame_idx].astype(np.float64)
        )
        deformed_mesh.compute_vertex_normals()

        vis.remove_geometry(arrow, reset_bounding_box=False)
        vis.remove_geometry(plane, reset_bounding_box=False)
        arrow, plane = get_arrow_and_plane_at_progress(
            centerline_src_lps, progress, bbox_diagonal
        )
        vis.add_geometry(arrow, reset_bounding_box=False)
        vis.add_geometry(plane, reset_bounding_box=False)

        vis.update_geometry(deformed_mesh)
        vis.update_geometry(centerline_src_set)
        for sphere in spheres_src:
            vis.update_geometry(sphere)
        vis.update_geometry(centerline_dst_set)
        for sphere in spheres_dst:
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
    parser = argparse.ArgumentParser(
        description="Preview a generated C3VD centerline warp animation export."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config used for generation")
    args = parser.parse_args()

    config = load_config(args.config)
    config_stem = Path(args.config).stem
    fps = float(config.get("fps", 30))

    output_dir = Path(str(config.get("output_root"))) / config_stem
    mesh, frame_vertices = load_export(output_dir)

    geometry = str(config["geometry"])
    reference_dir = resolve_reference_dir(config)
    centerline_path = resolve_centerline_path(config, geometry)

    cl_src = np.load(centerline_path).astype(float)
    cl_src_lps = ras_to_lps(cl_src)

    warped_centerline_name = f"centerline_warped_{config['transform_mode']}.npy"
    warped_centerline_path = output_dir / warped_centerline_name
    if not warped_centerline_path.exists():
        raise RuntimeError(f"Missing warped centerline: {warped_centerline_path}")

    cl_dst = np.load(warped_centerline_path).astype(float)
    cl_dst_lps = ras_to_lps(cl_dst)

    print(f"Previewing export: {output_dir}")
    print(f"Frames: {len(frame_vertices)}, fps={fps}")
    preview_animation(
        mesh,
        frame_vertices,
        cl_src_lps,
        cl_dst_lps,
        fps,
    )


if __name__ == "__main__":
    main()

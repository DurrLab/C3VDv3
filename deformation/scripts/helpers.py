import os

import numpy as np
import open3d as o3d


def ras_to_lps(points):
    out = points.copy()
    out[:, 0] *= -1
    out[:, 1] *= -1
    return out


def lps_to_ras(points):
    out = points.copy()
    out[:, 0] *= -1
    out[:, 1] *= -1
    return out


def resample_even_arc(points, n_out):
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or n_out <= 1:
        return points.copy()

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 1e-12:
        return np.repeat(points[:1], n_out, axis=0)

    s_new = np.linspace(0.0, total, n_out)
    out = np.zeros((n_out, 3), dtype=float)
    for i in range(3):
        out[:, i] = np.interp(s_new, s, points[:, i])
    return out


def preprocess_mesh_for_smoother_render(mesh, taubin_iterations=8, subdivision_iterations=0):
    original_v = len(mesh.vertices)
    original_f = len(mesh.triangles)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) > 0:
        bbox = mesh.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        merge_eps = max(diag * 1e-5, 1e-7)
        if hasattr(mesh, "merge_close_vertices"):
            mesh.merge_close_vertices(merge_eps)
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()

    if taubin_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iterations)

    if subdivision_iterations > 0:
        mesh = mesh.subdivide_loop(number_of_iterations=subdivision_iterations)

    mesh.compute_vertex_normals()
    print("Mesh preprocess:")
    print(f"  vertices: {original_v} -> {len(mesh.vertices)}")
    print(f"  triangles: {original_f} -> {len(mesh.triangles)}")
    return mesh


def write_topology_obj(path, reference_vertices, faces, reference_normals=None):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("# Topology mesh (fixed faces for all animation frames)\n")
        f.write("mtllib model.mtl\n")
        for v in reference_vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if reference_normals is not None and len(reference_normals) == len(reference_vertices):
            for n in reference_normals:
                f.write(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
            f.write("s 1\n")
            f.write("usemtl material_0\n")
            for tri in faces:
                i, j, k = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")
        else:
            f.write("s 1\n")
            f.write("usemtl material_0\n")
            for tri in faces:
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def compute_vertex_normals(vertices, triangles):
    normals = np.zeros_like(vertices)
    for tri in triangles:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        face_normal = np.cross(v1 - v0, v2 - v0)
        normals[i] += face_normal
        normals[j] += face_normal
        normals[k] += face_normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (normals / norms).astype(np.float32)


def project_vertices_to_centerline_arclength(vertices, centerline, s_vals, chunk_size=4096):
    seg_start = centerline[:-1]
    seg_end = centerline[1:]
    seg_vec = seg_end - seg_start
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1)
    seg_len2 = np.maximum(seg_len2, 1e-12)

    n_vertices = vertices.shape[0]
    vertex_s = np.empty(n_vertices, dtype=np.float64)

    for start_idx in range(0, n_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, n_vertices)
        v_chunk = vertices[start_idx:end_idx]

        diff = v_chunk[:, None, :] - seg_start[None, :, :]
        t = np.sum(diff * seg_vec[None, :, :], axis=2) / seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)

        proj = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
        d2 = np.sum((v_chunk[:, None, :] - proj) ** 2, axis=2)
        best_seg = np.argmin(d2, axis=1)

        row = np.arange(end_idx - start_idx)
        best_t = t[row, best_seg]
        vertex_s[start_idx:end_idx] = s_vals[best_seg] + best_t * (s_vals[best_seg + 1] - s_vals[best_seg])

    return vertex_s


def interpolate_centerline_points(centerline, s_vals, query_s):
    x = np.interp(query_s, s_vals, centerline[:, 0])
    y = np.interp(query_s, s_vals, centerline[:, 1])
    z = np.interp(query_s, s_vals, centerline[:, 2])
    return np.stack([x, y, z], axis=1)


def build_vertex_adjacency(n_vertices, triangles):
    neighbors = [set() for _ in range(n_vertices)]
    for tri in triangles:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[i].update((j, k))
        neighbors[j].update((i, k))
        neighbors[k].update((i, j))
    return [np.array(sorted(list(nbrs)), dtype=np.int32) for nbrs in neighbors]


def laplacian_smooth_scalar(values, neighbors, iterations=6, lamb=0.35):
    smoothed = values.astype(np.float64).copy()
    n = smoothed.shape[0]
    for _ in range(iterations):
        updated = smoothed.copy()
        for vid in range(n):
            nbr = neighbors[vid]
            if nbr.size == 0:
                continue
            nbr_mean = smoothed[nbr].mean()
            updated[vid] = (1.0 - lamb) * smoothed[vid] + lamb * nbr_mean
        smoothed = updated
    return smoothed

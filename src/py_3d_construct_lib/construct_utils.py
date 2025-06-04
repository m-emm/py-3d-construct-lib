import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from py_3d_construct_lib.spherical_tools import rotation_matrix_from_vectors


def normalize_edge(a, b):
    return tuple(sorted((a, b)))


def triangle_edges(tri):
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


def compute_triangle_normal(v0, v1, v2):
    return np.cross(v1 - v0, v2 - v0)


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def are_normals_similar(n1: np.ndarray, n2: np.ndarray, tol: float = 1e-3) -> bool:
    """
    Checks if two normals are nearly aligned (dot product close to 1.0).
    """
    n1 = normalize(n1)
    n2 = normalize(n2)
    return np.dot(n1, n2) > 1.0 - tol


def fibonacci_sphere(samples=100):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append(np.array([x, y, z]))
    return points


def compute_barycentric_coords(p, tri):
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return None  # degenerate triangle

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])


import math
from collections import Counter
from typing import Dict, List, Set, Tuple

Vertex = int
Triangle = Tuple[Vertex, Vertex, Vertex]
Edge = Tuple[Vertex, Vertex]
NewVertexMapping = Dict[Edge, Vertex]


def triangle_edges(tri: Triangle) -> List[Edge]:
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


def normalize_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def compute_area(p1, p2, p3):
    """Returns area of triangle with vertices p1, p2, p3 in 2D"""
    return abs(
        (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        / 2
    )


def rotate_triangle(tri: Tuple[int, int, int], k: int) -> Tuple[int, int, int]:
    return (tri[k % 3], tri[(k + 1) % 3], tri[(k + 2) % 3])


def split_triangle_topologically(tri, edge_to_new_vertex, perform_area_check=True):
    original_edges = [
        (tri[0], tri[1]),
        (tri[1], tri[2]),
        (tri[2], tri[0]),
    ]

    split_flags = [normalize_edge(*e) in edge_to_new_vertex for e in original_edges]

    # Find the rotation offset so that all split edges come first
    def count_split_flags(flags):  # how many split edges from the front
        count = 0
        for f in flags:
            if f:
                count += 1
            else:
                break
        return count

    best_offset = max(
        range(3), key=lambda k: count_split_flags(split_flags[k:] + split_flags[:k])
    )

    tri_rot = rotate_triangle(tri, best_offset)
    edge_rot = [
        (tri_rot[0], tri_rot[1]),
        (tri_rot[1], tri_rot[2]),
        (tri_rot[2], tri_rot[0]),
    ]

    # Assign local indices 0, 1, 2 to rotated triangle
    local_to_global = {0: tri_rot[0], 1: tri_rot[1], 2: tri_rot[2]}

    edge_to_local = {
        normalize_edge(0, 1): 3,
        normalize_edge(1, 2): 4,
        normalize_edge(2, 0): 5,
    }

    for local_edge, new_local_index in edge_to_local.items():
        # Map back to global edge
        global_edge = normalize_edge(
            local_to_global[local_edge[0]], local_to_global[local_edge[1]]
        )
        if global_edge in edge_to_new_vertex:
            v_new = edge_to_new_vertex[global_edge]
            local_to_global[new_local_index] = v_new

    # Determine case and return triangles as before
    num_splits = sum([normalize_edge(*e) in edge_to_new_vertex for e in edge_rot])

    CASE_TO_LOCAL_TRIANGLES = {
        0: [[0, 1, 2]],
        1: [[0, 3, 2], [3, 1, 2]],
        2: [[0, 3, 4], [3, 1, 4], [4, 2, 0]],
        3: [[0, 3, 5], [3, 4, 5], [3, 1, 4], [4, 2, 5]],
    }

    local_tris = CASE_TO_LOCAL_TRIANGLES[num_splits]
    final_tris = [[local_to_global[i] for i in tri] for tri in local_tris]

    # Optionally check area
    if perform_area_check:
        coords = {
            tri_rot[0]: (0.0, 0.0),
            tri_rot[1]: (1.0, 0.0),
            tri_rot[2]: (0.5, math.sqrt(3) / 2),
        }
        for i in range(3):
            a, b = edge_rot[i]
            canon = normalize_edge(a, b)
            if canon in edge_to_new_vertex:
                mid = edge_to_new_vertex[canon]
                pa = coords[a]
                pb = coords[b]
                coords[mid] = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

        original_area = compute_area(
            coords[tri_rot[0]], coords[tri_rot[1]], coords[tri_rot[2]]
        )
        new_area = sum(
            compute_area(coords[a], coords[b], coords[c]) for (a, b, c) in final_tris
        )
        if not math.isclose(original_area, new_area, rel_tol=1e-9):
            raise ValueError(f"Area mismatch: original {original_area}, new {new_area}")

    return final_tris


def compute_lay_flat_transform(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    Compute a 4×4 transformation matrix that lays the triangle (a, b, c) flat on the XY plane.
    The triangle will be rotated such that its normal aligns with +Z and translated so it lies at Z=0.

    Parameters:
        a, b, c: np.ndarray
            The 3D coordinates of the triangle vertices.

    Returns:
        A 4×4 np.ndarray representing the affine transform.
    """
    centroid = (a + b + c) / 3.0
    normal_vec = np.cross(b - a, c - a)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    target_normal = np.array([0.0, 0.0, 1.0])

    # Rotation matrix to align the normal to +Z
    R3 = rotation_matrix_from_vectors(normal_vec, target_normal)

    # Build the full transform: T_z * T(+centroid) * R * T(-centroid)
    T1 = np.eye(4)
    T1[:3, 3] = -centroid

    R4 = np.eye(4)
    R4[:3, :3] = R3

    T2 = np.eye(4)
    T2[:3, 3] = centroid

    M = T2 @ R4 @ T1

    # Apply to triangle and compute average z after transform
    pts_m = [(M @ np.hstack([v, 1.0]))[:3] for v in (a, b, c)]
    z_face = sum(p[2] for p in pts_m) / 3.0

    # Final shift to bring to Z=0
    T3 = np.eye(4)
    T3[2, 3] = -z_face

    return T3 @ M


@dataclass
class CylinderSpec:
    bottom: np.ndarray  # shape (3,)
    normal: np.ndarray  # shape (3,), must be normalized
    height: float
    radius: float

    def __post_init__(self):
        self.normal = self.normal / np.linalg.norm(self.normal)  # ensure unit


def intersect_edge_with_cylinder(p1, p2, cylinder: CylinderSpec, epsilon=1e-8):
    """
    Returns the (t1, t2) parameters along the edge p1→p2 where it enters/exits the cylinder.
    If no intersection, returns None.
    """
    from numpy.linalg import norm

    d = p2 - p1  # direction of the edge
    h = cylinder.normal / norm(cylinder.normal)  # normalized cylinder axis
    m = p1 - cylinder.bottom

    # Vector components orthogonal to cylinder axis
    d_perp = d - np.dot(d, h) * h
    m_perp = m - np.dot(m, h) * h

    A = np.dot(d_perp, d_perp)

    if A < epsilon:
        # Edge is parallel to axis; check if it's within radius
        dist_to_axis = np.linalg.norm(m_perp)
        if dist_to_axis > cylinder.radius + epsilon:
            return None  # Edge is outside the cylinder

        # Compute t values where edge enters/leaves via height
        t1 = (0.0 - np.dot(m, h)) / np.dot(d, h)
        t2 = (cylinder.height - np.dot(m, h)) / np.dot(d, h)

        t_enter = min(t1, t2)
        t_exit = max(t1, t2)

        if t_exit < 0 or t_enter > 1:
            return None

        return max(t_enter, 0.0), min(t_exit, 1.0)

    B = 2 * np.dot(d_perp, m_perp)
    C = np.dot(m_perp, m_perp) - cylinder.radius**2

    discriminant = B**2 - 4 * A * C

    if discriminant < -epsilon:
        return None  # no real roots, no intersection
    elif abs(discriminant) <= epsilon:
        discriminant = 0.0

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)

    t_enter = min(t1, t2)
    t_exit = max(t1, t2)

    # Clamp to edge segment
    if t_exit < 0 or t_enter > 1:
        return None

    return max(t_enter, 0.0), min(t_exit, 1.0)


def triangle_min_angle(p0, p1, p2):
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p0)
    c = np.linalg.norm(p0 - p1)
    angles = []
    for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
        cos_angle = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    return min(angles)

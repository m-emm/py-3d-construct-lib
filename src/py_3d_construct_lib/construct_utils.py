import math

import numpy as np


def normalize_edge(a, b):
    return tuple(sorted((a, b)))


def triangle_edges(tri):
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


def compute_triangle_normal(v0, v1, v2):
    return np.cross(v1 - v0, v2 - v0)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


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


def create_dodecahedron_geometry(radius=1.0):
    """
    Returns:
      verts: (20,3) numpy array of vertex coordinates on sphere of given radius
      faces: (12,5) numpy array of pentagon indices into verts (CCW)
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Create the 20 vertices
    raw_verts = np.array(
        [
            # 8 vertices at (±1, ±1, ±1)
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
            # 12 vertices at even permutations of (0, ±1/phi, ±phi)
            [0, -1 / phi, -phi],
            [0, -1 / phi, phi],
            [0, 1 / phi, -phi],
            [0, 1 / phi, phi],
            [-1 / phi, -phi, 0],
            [-1 / phi, phi, 0],
            [1 / phi, -phi, 0],
            [1 / phi, phi, 0],
            [-phi, 0, -1 / phi],
            [phi, 0, -1 / phi],
            [-phi, 0, 1 / phi],
            [phi, 0, 1 / phi],
        ],
        dtype=np.float64,
    )

    # Normalize to lie on sphere
    lengths = np.linalg.norm(raw_verts, axis=1)
    verts = raw_verts * (radius / lengths)[:, None]

    # Define the 12 pentagonal faces (indices into verts array)
    faces = np.array(
        [
            [0, 8, 10, 2, 16],
            [0, 16, 18, 1, 12],
            [0, 12, 13, 3, 8],
            [1, 18, 19, 5, 9],
            [1, 9, 11, 3, 13],
            [2, 10, 11, 9, 4],
            [2, 4, 17, 6, 16],
            [3, 11, 10, 8, 7],
            [3, 7, 15, 13, 12],
            [4, 14, 15, 7, 17],
            [4, 9, 5, 14, 17],
            [5, 19, 15, 14, 6],
        ],
        dtype=int,
    )

    return verts, faces

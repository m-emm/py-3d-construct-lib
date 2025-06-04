import numpy as np


def create_icosahedron_geometry(radius=1.0):
    """
    Returns:
      verts: (12,3) numpy array of vertex coordinates on sphere of given radius
      faces: (20,3) numpy array of triangle indices into verts
    """

    # golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 12 un-scaled verts in the "three.js" / Wikipedia order:
    raw_verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )

    # normalize so each lies on sphere of radius `radius`
    lengths = np.linalg.norm(raw_verts, axis=1)
    verts = raw_verts * (radius / lengths)[:, None]

    # the 20 faces (triangles), CCW when viewed from outside
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )

    return verts, faces


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


def create_cube_geometry(radius=1.0):
    """
    Returns:
      verts: (8,3) numpy array of vertex coordinates on sphere of given radius
      faces: (12,3) numpy array of triangle indices (2 per face, 6 faces)
    """
    # 8 cube corners
    raw_verts = np.array(
        [
            [-1, -1, -1],  # 0
            [1, -1, -1],  # 1
            [1, 1, -1],  # 2
            [-1, 1, -1],  # 3
            [-1, -1, 1],  # 4
            [1, -1, 1],  # 5
            [1, 1, 1],  # 6
            [-1, 1, 1],  # 7
        ],
        dtype=np.float64,
    )

    # normalize to radius
    verts = raw_verts / np.linalg.norm(raw_verts, axis=1)[:, None] * radius

    # triangles per face (CCW outward)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom (-Z)
            [4, 6, 5],
            [4, 7, 6],  # top (+Z)
            [0, 4, 5],
            [0, 5, 1],  # front (-Y)
            [1, 5, 6],
            [1, 6, 2],  # right (+X)
            [2, 6, 7],
            [2, 7, 3],  # back (+Y)
            [3, 7, 4],
            [3, 4, 0],  # left (-X)
        ],
        dtype=int,
    )

    return verts, faces


def create_tetrahedron_geometry(radius=1.0):
    """
    Returns:
      verts: (4,3) numpy array of vertex coordinates on sphere of given radius
      faces: (4,3) numpy array of triangle indices (all faces)
    """
    # Regular tetrahedron centered at origin
    raw_verts = np.array(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ],
        dtype=np.float64,
    )

    # normalize to sphere
    verts = raw_verts / np.linalg.norm(raw_verts, axis=1)[:, None] * radius

    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=int,
    )

    return verts, faces

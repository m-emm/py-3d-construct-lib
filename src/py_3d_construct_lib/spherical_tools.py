import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.spatial import Delaunay, cKDTree


# def cartesian_to_spherical_rotated(v: np.ndarray, center: np.ndarray):
#     v = v - center
#     x, y, z = v
#     # rotate to local spherical frame
#     x_, y_, z_ = x, z, y
#     r = np.linalg.norm([x_, y_, z_])
#     theta = np.arccos(z_ / r)
#     phi = np.arctan2(y_, x_)
#     return (r, theta, phi)


# def spherical_to_cartesian_rotated(sph: tuple, center: np.ndarray, radius_offset=0):
#     r, theta, phi = sph
#     r += radius_offset
#     x_ = r * np.cos(phi) * np.sin(theta)
#     y_ = r * np.sin(phi) * np.sin(theta)
#     z_ = r * np.cos(theta)
#     # rotate back to global frame
#     x, y, z = x_, z_, y_
#     return np.array([x, y, z]) + center


def spherical_to_cartesian_jackson(sph: tuple, radius_offset=0, sphere_center=None):
    r, theta, phi = sph
    r += radius_offset
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    if sphere_center is not None:
        x += sphere_center[0]
        y += sphere_center[1]
        z += sphere_center[2]
    return np.array([x, y, z])


def cartesian_to_spherical_jackson(v: np.ndarray, center=None):
    if center is not None:
        v = v - center
    x, y, z = v
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return (r, theta, phi)


def shrink_triangle(A, B, C, border_width, epsilon=1e-6):
    def compute_offset_point(p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < epsilon or norm2 < epsilon:
            raise ValueError("Degenerate triangle corner with zero-length edge.")

        v1n = v1 / norm1
        v2n = v2 / norm2

        dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
        angle = np.arccos(dot)
        sin_half_angle = np.sin(angle / 2)

        if sin_half_angle < epsilon:
            raise ValueError(
                f"Corner too sharp (angle={np.degrees(angle):.2f}°), cannot safely shrink."
            )

        offset_length = border_width / sin_half_angle

        # Require offset to be smaller than both adjacent edge lengths
        if offset_length > min(norm1, norm2):
            raise ValueError(
                f"Offset {offset_length:.4f} too large for triangle at corner with edge lengths "
                f"{norm1:.4f} and {norm2:.4f}."
            )

        bisector = v1n + v2n
        bisector /= np.linalg.norm(bisector)
        return p0 + bisector * offset_length

    A_new = compute_offset_point(A, B, C)
    B_new = compute_offset_point(B, C, A)
    C_new = compute_offset_point(C, A, B)

    return A_new, B_new, C_new


def create_shell_triangle_geometry(
    triangle_spherical_vertexes,
    sphere_center,
    shell_thickness,
    shrinkage=0.1,
    shrink_border=0,
):

    if len(triangle_spherical_vertexes) != 3:
        raise ValueError("triangle_spherical_vertexes must have 3 elements")

    for i in range(3):
        if len(triangle_spherical_vertexes[i]) != 3:
            raise ValueError(
                "Each element of triangle_spherical_vertexes must have 3 elements (r, theta, phi)"
            )

    cartesian_vertexes = [
        spherical_to_cartesian_jackson(v, sphere_center=sphere_center)
        for v in triangle_spherical_vertexes
    ]
    outside_cartesian_vertexes = [
        spherical_to_cartesian_jackson(
            v, radius_offset=shell_thickness, sphere_center=sphere_center
        )
        for v in triangle_spherical_vertexes
    ]

    # check if the vertexes are in the right order
    # if not, reverse the order

    if (
        np.cross(
            cartesian_vertexes[1] - cartesian_vertexes[0],
            cartesian_vertexes[2] - cartesian_vertexes[0],
        )[2]
        < 0
    ):
        cartesian_vertexes[1], cartesian_vertexes[2] = (
            cartesian_vertexes[2],
            cartesian_vertexes[1],
        )
        outside_cartesian_vertexes[1], outside_cartesian_vertexes[2] = (
            outside_cartesian_vertexes[2],
            outside_cartesian_vertexes[1],
        )

    centroid = np.sum(cartesian_vertexes, axis=0) / 6
    centroid += np.sum(outside_cartesian_vertexes, axis=0) / 6

    for i in range(3):
        cartesian_vertexes[i] = cartesian_vertexes[i] - shrinkage * (
            cartesian_vertexes[i] - centroid
        )
        outside_cartesian_vertexes[i] = outside_cartesian_vertexes[i] - shrinkage * (
            outside_cartesian_vertexes[i] - centroid
        )

    # shrink with border
    if shrink_border > 0:
        cartesian_vertexes[0], cartesian_vertexes[1], cartesian_vertexes[2] = (
            shrink_triangle(
                cartesian_vertexes[0],
                cartesian_vertexes[1],
                cartesian_vertexes[2],
                border_width=shrink_border,
            )
        )
        (
            outside_cartesian_vertexes[0],
            outside_cartesian_vertexes[1],
            outside_cartesian_vertexes[2],
        ) = shrink_triangle(
            outside_cartesian_vertexes[0],
            outside_cartesian_vertexes[1],
            outside_cartesian_vertexes[2],
            border_width=shrink_border,
        )

    # Now use these six points to define a prism 
    vertexes = {i: v for i, v in enumerate(cartesian_vertexes)}
    outside_vertexes = {i + 3: v for i, v in enumerate(outside_cartesian_vertexes)}
    cartesian_vertexes = {**vertexes, **outside_vertexes}
    maps = {
        "vertexes": cartesian_vertexes,
        "faces": {
            0: [0, 2, 1],  # bottom
            1: [3, 4, 5],  # top
            2: [0, 1, 4],
            3: [0, 4, 3],
            4: [1, 2, 5],
            5: [1, 5, 4],
            6: [2, 0, 3],
            7: [2, 3, 5],
        },
    }

    return maps


def ray_triangle_intersect(ray_origin, ray_vector, triangle):
    EPSILON = 1e-8
    vertex0, vertex1, vertex2 = triangle
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return None  # Ray is parallel to triangle
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > EPSILON:
        return ray_origin + ray_vector * t  # Intersection point
    else:
        return None  # Line intersects but not the ray


import numpy as np
from scipy.spatial import ConvexHull


def is_inside_convex_polygon_2d(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Test if each point in `points` lies inside the convex polygon defined by `polygon`.

    Parameters:
        polygon: (N, 2) array of 2D points (ordered, counterclockwise)
        points:  (M, 2) array of 2D test points

    Returns:
        mask: (M,) boolean array, True if point is inside the polygon
    """
    n_edges = polygon.shape[0]
    n_points = points.shape[0]

    inside = np.ones(n_points, dtype=bool)

    for i in range(n_edges):
        a = polygon[i]
        b = polygon[(i + 1) % n_edges]
        edge = b - a
        to_point = points - a  # (M, 2)

        # 2D cross product: edge_x * point_y - edge_y * point_x
        cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]

        # For CCW polygon: point must be on the left side of edge (cross >= 0)
        inside &= cross >= 0

    return inside


def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def cartesian_to_spherical(xyz):
    x, y, z = xyz.T
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.stack([theta, phi], axis=1)


def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s**2))
    return R


def azimuthal_projection(theta_phi, extension=0):
    theta, phi = theta_phi.T
    theta = theta + extension
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    return np.stack([x, y], axis=1)


def filter_outside_spherical_cap(cap_theta_phi, other_theta_phi, border_extension=0):
    """
    Filters points outside a spherical cap region defined by `cap_theta_phi`.

    Parameters:
        cap_theta_phi: (N, 2) array of [theta, phi] in radians
        other_theta_phi: (M, 2) array of [theta, phi] in radians

    Returns:
        (K, 2) array of [theta, phi] points from `other_theta_phi` that are outside the cap.
    """
    cap_xyz = spherical_to_cartesian(cap_theta_phi[:, 0], cap_theta_phi[:, 1])
    center_vec = cap_xyz.mean(axis=0)
    center_vec /= np.linalg.norm(center_vec)

    R = rotation_matrix_from_vectors(center_vec, np.array([0, 0, 1]))

    cap_rotated = cap_xyz @ R.T
    cap_rotated_theta_phi = cartesian_to_spherical(cap_rotated)

    # project rotated cap to plane
    cap_proj = azimuthal_projection(cap_rotated_theta_phi, extension=border_extension)

    hull = ConvexHull(cap_proj)

    hull_vertices = cap_proj[hull.vertices]

    # rotate and project other points
    other_xyz = spherical_to_cartesian(other_theta_phi[:, 0], other_theta_phi[:, 1])
    other_rotated = other_xyz @ R.T
    other_rotated_theta_phi = cartesian_to_spherical(other_rotated)
    other_proj = azimuthal_projection(other_rotated_theta_phi)

    mask_inside = is_inside_convex_polygon_2d(hull_vertices, other_proj)

    mask_outside = ~mask_inside

    return other_theta_phi[mask_outside], mask_outside, hull.vertices


def coordinate_system_transform(origin_a, up_a, out_a, origin_b, up_b, out_b):
    """
    Compute the rigid transformation (rotation and translation) needed
    to align coordinate system A to coordinate system B in 3D space.

    Each coordinate system is defined by:
      - an origin point,
      - two orthogonal direction vectors: "up" and "out".

    The third basis vector ("along") is computed as the cross product
    of "up" and "out". All three vectors define a right-handed orthonormal basis.

    The transformation brings the "up", "out", and "along" vectors of system A
    into alignment with those of system B, and translates the origin from A to B.

    Parameters:
    ----------
    origin_a : array-like, shape (3,)
        Origin of the source coordinate system A.
    up_a : array-like, shape (3,)
        "Up" vector of coordinate system A (must be orthogonal to out_a).
    out_a : array-like, shape (3,)
        "Out" vector of coordinate system A (must be orthogonal to up_a).

    origin_b : array-like, shape (3,)
        Origin of the target coordinate system B.
    up_b : array-like, shape (3,)
        "Up" vector of coordinate system B (must be orthogonal to out_b).
    out_b : array-like, shape (3,)
        "Out" vector of coordinate system B (must be orthogonal to up_b).

    Returns:
    -------
    transform : dict
        A dictionary with the following keys:
        - "rotation_axis": tuple of 3 floats representing the axis of rotation.
        - "rotation_angle": float, angle in radians to rotate around the axis.
        - "translation": tuple of 3 floats, the translation vector from origin_a to origin_b.

    Notes:
    -----
    - All direction vectors are normalized internally.
    - The rotation is defined as rotating around `origin_a` by the given axis and angle.
    - The translation is applied after the rotation to align the origins.

    Example:
    --------
    >>> transform = coordinate_system_transform(
    ...     origin_a=[0,0,0], up_a=[0,0,1], out_a=[1,0,0],
    ...     origin_b=[1,2,3], up_b=[0,1,0], out_b=[0,0,1])
    >>> transform["rotation_axis"]
    >>> transform["rotation_angle"]
    >>> transform["translation"]
    """

    # Convert inputs to numpy arrays
    origin_a = np.asarray(origin_a, dtype=float)
    origin_b = np.asarray(origin_b, dtype=float)
    up_a = np.asarray(up_a, dtype=float) / np.linalg.norm(up_a)
    out_a = np.asarray(out_a, dtype=float) / np.linalg.norm(out_a)
    up_b = np.asarray(up_b, dtype=float) / np.linalg.norm(up_b)
    out_b = np.asarray(out_b, dtype=float) / np.linalg.norm(out_b)

    # Build full orthonormal bases
    along_a = np.cross(up_a, out_a)
    R_a = np.column_stack([up_a, out_a, along_a])

    along_b = np.cross(up_b, out_b)
    R_b = np.column_stack([up_b, out_b, along_b])

    # Rotation matrix to go from A to B
    R = R_b @ R_a.T

    # Extract axis-angle
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if np.isclose(angle, 0):
        axis = np.array([1, 0, 0])  # arbitrary
    elif np.isclose(angle, np.pi):
        # Use eigenvector of R with eigenvalue 1
        eigvals, eigvecs = np.linalg.eigh(R)
        axis = eigvecs[:, np.isclose(eigvals, 1)].flatten()
        axis /= np.linalg.norm(axis)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * np.sin(angle)
        )

    # Translation vector
    translation = origin_b - origin_a

    return {
        "rotation_angle": float(angle),
        "rotation_axis": tuple(axis),
        "translation": tuple(translation),
    }

    
    


def coordinate_system_transform_to_matrix(transform: dict) -> np.ndarray:
    angle = transform["rotation_angle"]
    axis = np.array(transform["rotation_axis"])
    translation = np.array(transform["translation"])

    # Build 3x3 rotation matrix from axis/angle
    R = rotation_matrix_from_vectors(axis, axis)  # identity if angle == 0
    if angle != 0:
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = translation
    return A
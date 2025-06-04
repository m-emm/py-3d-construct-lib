import numpy as np
from py_3d_construct_lib.spherical_tools import (
    cartesian_to_spherical_jackson,
    coordinate_system_transform,
    create_shell_triangle_geometry,
    matrix_to_coordinate_system_transform,
    ray_triangle_intersect,
    rotation_matrix_from_vectors,
    spherical_to_cartesian_jackson,
)


def test_cartesian_to_spherical_jackson():
    np.random.seed(42)  # For reproducibility
    size = 2000

    for _ in range(100):
        x, y, z = np.random.uniform(-size, size, 3)
        r, theta, phi = cartesian_to_spherical_jackson((x, y, z))

        xyz = spherical_to_cartesian_jackson((r, theta, phi))
        assert np.allclose((x, y, z), xyz, atol=1e-6), f"Failed for input: {(x, y, z)}"

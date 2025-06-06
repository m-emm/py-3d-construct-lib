import numpy as np
from py_3d_construct_lib.geometries import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
    create_icosahedron_geometry,
    create_tetrahedron_geometry,
)
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)


def test_create_cube_geometry():
    points, faces = create_cube_geometry(1.0)
    assert len(points) == 8
    assert len(faces) == 12  # triangles
    assert all(len(face) == 3 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh(points, faces)


def test_create_tetrahedron_geometry():
    points, faces = create_tetrahedron_geometry(1.0)
    assert len(points) == 4
    assert len(faces) == 4  # triangles
    assert all(len(face) == 3 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh(points, faces)


def test_create_dodecahedron_geometry():

    points, faces = create_dodecahedron_geometry(1.0)
    assert len(points) == 20
    assert len(faces) == 12  # triangles
    assert all(len(face) == 5 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh.from_point_cloud(points)


def test_create_icosahedron_geometry():
    points, faces = create_icosahedron_geometry(1.0)
    assert len(points) == 12
    assert len(faces) == 20  # triangles
    assert all(len(face) == 3 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh(points, faces)


def test_create_fibonacci_sphere_geometry():
    points, faces = create_fibonacci_sphere_geometry(1.0, samples=100)

    assert len(points) == 100
    assert all(len(p) == 3 for p in points)

    # Check that the points are uniformly distributed on the sphere
    norms = [np.linalg.norm(p) for p in points]
    assert all(
        np.isclose(norm, 1.0) for norm in norms
    )  # All points should be on the unit sphere

    _ = PartitionableSpheroidTriangleMesh(points, faces)

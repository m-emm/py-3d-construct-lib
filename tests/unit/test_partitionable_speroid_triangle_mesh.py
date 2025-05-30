import numpy as np
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
    TransformedRegionView,
)


def test_split_region_by_cap():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition_0 = mesh.get_trivial_partition()

    partition = partition_0.split_region_by_cap(
        0, initial_seed_triangle_index=0, target_area_fraction=0.4
    )


def test_split_region_by_polar_oriented_plane():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition = mesh.get_trivial_partition()

    partition_2 = partition.split_region_by_polar_oriented_plane(
        region_id=0, target_area_fraction=0.5, phi=np.pi / 4
    )


def test_split_twice():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition_0 = mesh.get_trivial_partition()

    partition = partition_0.split_region_by_cap(
        0, initial_seed_triangle_index=0, target_area_fraction=0.4
    )

    print(f"***Partition after first split: {partition}")

    partition_2 = partition.split_region_by_polar_oriented_plane(
        region_id=1, target_area_fraction=0.5, phi=np.pi / 4
    )


def test_split_top_bottom_caps():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition_0 = mesh.get_trivial_partition()

    trv = TransformedRegionView(partition_0, region_id=0)

    V, F, E = trv.get_transformed_vertices_faces_boundary_edges()

    # find the poles

    top_pole_vertex_index = np.argmax(V[:, 2])
    bottom_pole_vertex_index = np.argmin(V[:, 2])
    print(
        f"Top pole vertex index: {top_pole_vertex_index}, Bottom pole vertex index: {bottom_pole_vertex_index}"
    )

    # find a  triangle each that contains the top and bottom poles
    top_pole_triangle_index = np.where(np.isin(F, top_pole_vertex_index))[0][0]
    bottom_pole_triangle_index = np.where(np.isin(F, bottom_pole_vertex_index))[0][0]

    print(
        f"Top pole triangle index: {top_pole_triangle_index}, Bottom pole triangle index: {bottom_pole_triangle_index}"
    )

    partition = partition_0.split_region_by_cap(
        0, initial_seed_triangle_index=top_pole_triangle_index, target_area_fraction=0.1
    )

    partition_2 = partition.split_region_by_cap(
        0,
        initial_seed_triangle_index=bottom_pole_triangle_index,
        target_area_fraction=0.1,
    )

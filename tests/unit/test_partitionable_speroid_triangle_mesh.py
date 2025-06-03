import logging
from collections import defaultdict

import numpy as np
from py_3d_construct_lib.construct_utils import (
    compute_triangle_normal,
    create_dodecahedron_geometry,
    normalize,
    triangle_area,
)
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from py_3d_construct_lib.transformed_region_view import TransformedRegionView

_logger = logging.getLogger(__name__)


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


def test_add_vertex_on_edge_of_tetrahedron():
    # Define tetrahedron vertices
    vertices = np.array(
        [
            [1, 1, 1],  # 0
            [-1, -1, 1],  # 1
            [-1, 1, -1],  # 2
            [1, -1, -1],  # 3
        ]
    )

    # Define faces (each face is a triangle)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
    )

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces)

    # Pick triangle 0: [0, 1, 2], and place new vertex on edge (0, 1)
    barycentric_coords = [0.5, 0.5, 0.0]

    new_mesh = mesh.add_vertex_in_face(0, barycentric_coords)

    # Check vertex and face counts
    assert len(new_mesh.vertices) == 5
    assert len(new_mesh.faces) == 6

    # Check that the new vertex lies on the edge between vertex 0 and 1
    expected = 0.5 * mesh.vertices[0] + 0.5 * mesh.vertices[1]
    actual = new_mesh.vertices[-1]
    assert np.allclose(actual, expected)

    # Build directed edge map
    directed_edge_count = defaultdict(int)
    for face in new_mesh.faces:
        for i in range(3):
            a = face[i]
            b = face[(i + 1) % 3]
            directed_edge_count[(a, b)] += 1

    # For every directed edge (a → b), its reverse (b → a) must also exist once
    for (a, b), count in directed_edge_count.items():
        reverse_count = directed_edge_count.get((b, a), 0)
        assert count == 1, f"Edge ({a} → {b}) appears {count} times"
        assert reverse_count == 1, f"Reverse edge ({b} → {a}) missing or not balanced"


def test_add_vertex_inside_triangle_of_tetrahedron():
    # Define tetrahedron vertices
    vertices = np.array(
        [
            [1, 1, 1],  # 0
            [-1, -1, 1],  # 1
            [-1, 1, -1],  # 2
            [1, -1, -1],  # 3
        ]
    )

    # Define faces (each face is a triangle)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
    )

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces)

    # Pick triangle 0: [0, 1, 2], and place new vertex strictly inside
    barycentric_coords = [1 / 3, 1 / 3, 1 / 3]  # inside the triangle

    new_mesh = mesh.add_vertex_in_face(0, barycentric_coords)

    # Check vertex and face counts
    assert len(new_mesh.vertices) == 5
    assert len(new_mesh.faces) == 6  # +2 because 1 replaced by 3

    # Check that the new vertex lies at the expected position
    expected = (
        (1 / 3) * mesh.vertices[0]
        + (1 / 3) * mesh.vertices[1]
        + (1 / 3) * mesh.vertices[2]
    )
    actual = new_mesh.vertices[-1]
    assert np.allclose(actual, expected)

    # Check all directed edges occur exactly once, and reversed ones too
    directed_edge_count = defaultdict(int)
    for face in new_mesh.faces:
        for i in range(3):
            a = face[i]
            b = face[(i + 1) % 3]
            directed_edge_count[(a, b)] += 1

    for (a, b), count in directed_edge_count.items():
        reverse_count = directed_edge_count.get((b, a), 0)
        assert count == 1, f"Edge ({a} → {b}) appears {count} times"
        assert reverse_count == 1, f"Reverse edge ({b} → {a}) missing or not balanced"


def test_perforate_along_plane_tetrahedron():
    # Define tetrahedron vertices
    vertices = np.array(
        [
            [1, 1, 1],  # 0
            [-1, -1, 1],  # 1
            [-1, 1, -1],  # 2
            [1, -1, -1],  # 3
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
    )
    labels = ["A", "B", "C", "D"]

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces, vertex_labels=labels)

    # Choose a plane that cuts diagonally through the tetrahedron:
    # Plane x = 0 should intersect the edges AB, AC, AD (i.e., vertex 0 to others)
    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])

    # Act
    cut_mesh, _ = mesh.perforate_along_plane(plane_origin, plane_normal)

    _logger.info(f"Cut mesh labels: {cut_mesh.vertex_labels}")

    # We expect edges [0,1], [0,2] to be cut => 2 new vertices
    expected_cut_edges = [("A", "B"), ("A", "C")]
    for pair in expected_cut_edges:
        label1 = f"{pair[0]}__{pair[1]}"
        label2 = f"{pair[1]}__{pair[0]}"
        candidates = cut_mesh.get_vertices_by_label(label1)
        if not candidates:
            candidates = cut_mesh.get_vertices_by_label(label2)
        assert (
            len(candidates) == 1
        ), f"Expected one vertex for edge {pair}, got {candidates}"


def test_perforate_along_plane_tetrahedron_partial():
    # Define tetrahedron vertices
    vertices = np.array(
        [
            [1, 1, 1],  # 0
            [-1, -1, 1],  # 1
            [-1, 1, -1],  # 2
            [1, -1, -1],  # 3
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],  # face 0
            [0, 3, 1],  # face 1
            [0, 2, 3],  # face 2
            [1, 3, 2],  # face 3
        ]
    )
    labels = ["A", "B", "C", "D"]

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces, vertex_labels=labels)

    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])

    # Only perforate triangle 0
    triangle_indices = [0]
    cut_mesh, face_index_map = mesh.perforate_along_plane(
        plane_origin, plane_normal, epsilon=1e-9, triangle_indices=triangle_indices
    )

    # Check that new vertices are inserted for edges [0,1] and [0,2], but not for others
    expected_cut_labels = {"A__B", "A__C", "B__A", "C__A"}
    found_cut_labels = set(l for l in cut_mesh.vertex_labels if "__" in l)
    assert (
        found_cut_labels <= expected_cut_labels
    ), f"Unexpected cut labels: {found_cut_labels}"

    # Ensure expected faces are subdivided
    old_faces_per_subdivided = {k: v for k, v in face_index_map.items() if len(v) > 1}
    expected_split_faces = {0, 1, 2}
    assert (
        set(old_faces_per_subdivided.keys()) == expected_split_faces
    ), f"Expected faces {expected_split_faces} to be split, got {old_faces_per_subdivided}"
    # Faces that should be split due to shared edges with triangle 0
    expected_split_faces = {0, 1, 2}
    for i in range(4):
        if i in expected_split_faces:
            assert len(face_index_map[i]) > 1, f"Face {i} should be split"
        else:
            assert len(face_index_map[i]) == 1, f"Face {i} should not be split"

    # Confirm labels exist for both new cut edges
    for a, b in [("A", "B"), ("A", "C")]:
        candidates = cut_mesh.get_vertices_by_label(f"{a}__{b}")
        if not candidates:
            candidates = cut_mesh.get_vertices_by_label(f"{b}__{a}")
        assert len(candidates) == 1, f"Missing or duplicate cut vertex on edge {a}-{b}"

    _logger.info(
        f"Cut mesh: {len(cut_mesh.vertices)} vertices, {len(cut_mesh.faces)} faces"
    )
    _logger.info(f"Cut labels: {cut_mesh.vertex_labels}")


def test_perforate_along_plane_dodecahedron():

    points, _ = create_dodecahedron_geometry(1.0)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    # Choose a plane that cuts diagonally through the tetrahedron:
    # Plane x = 0 should intersect the edges AB, AC, AD (i.e., vertex 0 to others)
    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])

    # Act
    cut_mesh, _ = mesh.perforate_along_plane(plane_origin, plane_normal)

    _logger.info(f"Cut mesh labels: {cut_mesh.vertex_labels}")
    assert cut_mesh.vertex_labels == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "5__12",
        "12__14",
        "0__4",
        "0__14",
        "6__13",
        "13__15",
    ]


def test_perforate_along_plane_dodecahedron_check_normals():

    def check_normals_point_outward(mesh: PartitionableSpheroidTriangleMesh, label=""):
        total_area = 0.0
        inward_count = 0
        outward_count = 0
        zero_normal_count = 0

        for i, face in enumerate(mesh.faces):
            v0, v1, v2 = [mesh.vertices[vi] for vi in face]
            normal = compute_triangle_normal(v0, v1, v2)
            center = (v0 + v1 + v2) / 3.0
            center_dir = normalize(center)

            dot = np.dot(normal, center_dir)
            area = triangle_area(v0, v1, v2)
            total_area += area

            if dot < -1e-4:
                inward_count += 1
                _logger.warning(f"{label} Triangle {i} points inward (dot={dot:.4f})")
            elif dot > 1e-4:
                outward_count += 1
            else:
                zero_normal_count += 1
                _logger.warning(
                    f"{label} Triangle {i} has degenerate normal (dot={dot:.4f})"
                )

        _logger.info(
            f"{label} Triangle normals: outward={outward_count}, inward={inward_count}, degenerate={zero_normal_count}, total_area={total_area:.6f}"
        )
        return total_area

    points, _ = create_dodecahedron_geometry(1.0)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    area_before = check_normals_point_outward(mesh, label="BEFORE")

    # Cut it
    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 1.0, 0.0])
    cut_mesh, _ = mesh.perforate_along_plane(plane_origin, plane_normal)

    area_after = check_normals_point_outward(cut_mesh, label="AFTER")

    assert (
        abs(area_before - area_after) < 1e-5
    ), f"Area mismatch: before={area_before}, after={area_after}"

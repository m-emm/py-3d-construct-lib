import logging
from collections import defaultdict

import numpy as np
from py_3d_construct_lib.connector_hint import ConnectorHint
from py_3d_construct_lib.construct_utils import create_dodecahedron_geometry
from py_3d_construct_lib.mesh_partition import merge_collinear_connectors
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from py_3d_construct_lib.transformed_region_view import TransformedRegionView

_logger = logging.getLogger(__name__)


def test_perforated():
    points, _ = create_dodecahedron_geometry(1.0)

    # Step 1: Create the initial mesh and trivial partition (one region)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = mesh.get_trivial_partition()

    # Step 2: Define a plane (cut through origin, normal in x-direction)
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])  # vertical yz-plane

    # Step 3: Perforate and split
    new_partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    # Step 4: Analyze the new face-to-region mapping
    regions = defaultdict(list)
    for face_idx, region_id in new_partition.face_to_region_map.items():
        regions[region_id].append(face_idx)

    # Step 5: Check that we got two distinct regions
    assert len(regions) == 2, f"Expected 2 regions, got {len(regions)}"
    sizes = {rid: len(faces) for rid, faces in regions.items()}
    _logger.info(f"Perforated region sizes: {sizes}")

    # Step 6: Check that no triangle was lost
    total_faces = sum(sizes.values())
    assert total_faces == len(
        new_partition.mesh.faces
    ), f"Expected {len(new_partition.mesh.faces)} faces assigned, got {total_faces}"

    # Optional: check spatial separation using centroids
    centroids_by_region = {
        rid: np.array(
            [
                new_partition.mesh.vertices[new_partition.mesh.faces[f]].mean(axis=0)
                for f in face_indices
            ]
        )
        for rid, face_indices in regions.items()
    }

    for rid, centroids in centroids_by_region.items():
        avg = centroids.mean(axis=0)
        _logger.info(f"Region {rid} avg centroid: {avg}")


def test_merge_two_collinear_connector_hints():
    # Define two connector hints with a common endpoint
    # They should be merged into a single connector
    a1 = np.array([0.0, 0.0, 0.0])
    b1 = np.array([1.0, 0.0, 0.0])
    b2 = np.array([2.0, 0.0, 0.0])

    # Triangle normals are identical
    normal = np.array([0.0, 0.0, 1.0])

    # Apex vertices not on the edge
    apex_a1 = np.array([0.5, 1.0, 0.0])
    apex_a2 = np.array([1.5, 1.0, 0.0])
    apex_b1 = np.array([0.5, -1.0, 0.0])
    apex_b2 = np.array([1.5, -1.0, 0.0])

    ch1 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a1, b1, apex_a1),
        triangle_b_vertices=(b1, a1, apex_b1),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=b1 - a1,
        edge_centroid=(a1 + b1) / 2,
        original_edges=[],
        face_pair_ids=[(1, 1)],
        start_vertex=a1,
        end_vertex=b1,
    )

    ch2 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(b1, b2, apex_a2),
        triangle_b_vertices=(b2, b1, apex_b2),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=b2 - b1,
        edge_centroid=(b1 + b2) / 2,
        original_edges=[],
        face_pair_ids=[(2, 2)],
        start_vertex=b1,
        end_vertex=b2,
    )

    result = merge_collinear_connectors([ch1, ch2])

    # Expect one merged hint
    assert len(result) == 1, f"Expected 1 merged connector hint, got {len(result)}"

    merged = result[0]
    expected_vec = b2 - a1
    assert np.allclose(
        merged.edge_vector, expected_vec / np.linalg.norm(expected_vec)
    ), f"Expected edge_vector to be normalized {expected_vec}, got {merged.edge_vector}"

    expected_mid = (a1 + b2) / 2
    assert np.allclose(
        merged.edge_centroid, expected_mid
    ), f"Expected edge_centroid {expected_mid}, got {merged.edge_centroid}"

    # Vertex chain must match start and end
    tri_a = merged.triangle_a_vertices
    tri_b = merged.triangle_b_vertices
    assert np.allclose(tri_a[0], a1), "Merged triangle_a should start at a1"
    assert np.allclose(tri_a[1], b2), "Merged triangle_a should end at b2"
    assert np.allclose(tri_b[0], b2), "Merged triangle_b should start at b2"
    assert np.allclose(tri_b[1], a1), "Merged triangle_b should end at a1"

    # Check that face IDs were merged
    assert sorted(merged.face_pair_ids) == [
        (1, 1),
        (2, 2),
    ], "face_pair_ids not merged properly"

    print("test_merge_two_collinear_connector_hints passed.")


def test_merge_three_collinear_connector_hints():
    a0 = np.array([-1.0, 0.0, 0.0])
    a1 = np.array([0.0, 0.0, 0.0])
    a2 = np.array([1.0, 0.0, 0.0])
    a3 = np.array([2.0, 0.0, 0.0])

    normal = np.array([0.0, 0.0, 1.0])

    apex_a1 = np.array([-0.5, 1.0, 0.0])
    apex_a2 = np.array([0.5, 1.0, 0.0])
    apex_a3 = np.array([1.5, 1.0, 0.0])
    apex_b1 = np.array([-0.5, -1.0, 0.0])
    apex_b2 = np.array([0.5, -1.0, 0.0])
    apex_b3 = np.array([1.5, -1.0, 0.0])

    ch1 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a0, a1, apex_a1),
        triangle_b_vertices=(a1, a0, apex_b1),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=a1 - a0,
        edge_centroid=(a0 + a1) / 2,
        original_edges=[],
        face_pair_ids=[(1, 1)],
        start_vertex=a0,
        end_vertex=a1,
    )

    ch2 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a1, a2, apex_a2),
        triangle_b_vertices=(a2, a1, apex_b2),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=a2 - a1,
        edge_centroid=(a1 + a2) / 2,
        original_edges=[],
        face_pair_ids=[(2, 2)],
        start_vertex=a1,
        end_vertex=a2,
    )

    ch3 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a2, a3, apex_a3),
        triangle_b_vertices=(a3, a2, apex_b3),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=a3 - a2,
        edge_centroid=(a2 + a3) / 2,
        original_edges=[],
        face_pair_ids=[(3, 3)],
        start_vertex=a2,
        end_vertex=a3,
    )

    result = merge_collinear_connectors([ch1, ch2, ch3])
    assert len(result) == 1, f"Expected 1 merged connector hint, got {len(result)}"
    merged = result[0]

    expected_vec = a3 - a0
    expected_mid = (a0 + a3) / 2

    assert np.allclose(
        merged.edge_vector, expected_vec / np.linalg.norm(expected_vec)
    ), f"Expected edge_vector {expected_vec}, got {merged.edge_vector}"
    assert np.allclose(
        merged.edge_centroid, expected_mid
    ), f"Expected edge_centroid {expected_mid}, got {merged.edge_centroid}"

    tri_a = merged.triangle_a_vertices
    tri_b = merged.triangle_b_vertices
    assert np.allclose(tri_a[0], a0), "Merged triangle_a should start at a0"
    assert np.allclose(tri_a[1], a3), "Merged triangle_a should end at a3"
    assert np.allclose(tri_b[0], a3), "Merged triangle_b should start at a3"
    assert np.allclose(tri_b[1], a0), "Merged triangle_b should end at a0"
    assert sorted(merged.face_pair_ids) == [
        (1, 1),
        (2, 2),
        (3, 3),
    ], "face_pair_ids not merged properly"

    print("test_merge_three_collinear_connector_hints passed.")


def make_connector_hint(start, end, fid):
    normal = np.array([0.0, 0.0, 1.0])
    apex_a = (start + end) / 2 + np.array([0.0, 1.0, 0.0])
    apex_b = (start + end) / 2 + np.array([0.0, -1.0, 0.0])
    return ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(start, end, apex_a),
        triangle_b_vertices=(end, start, apex_b),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=end - start,
        edge_centroid=(start + end) / 2,
        original_edges=[],
        face_pair_ids=[(fid, fid)],
        start_vertex=start,
        end_vertex=end,
    )


def test_merge_three_collinear_connector_hints_backward():
    # This time we list the hints in reverse order to force backward chaining
    hints = [
        make_connector_hint(
            start=np.array([1.0, 0.0, 0.0]), end=np.array([2.0, 0.0, 0.0]), fid=3
        ),
        make_connector_hint(
            start=np.array([0.0, 0.0, 0.0]), end=np.array([1.0, 0.0, 0.0]), fid=2
        ),
        make_connector_hint(
            start=np.array([-1.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 0.0]), fid=1
        ),
    ]

    merged = merge_collinear_connectors(hints)
    assert len(merged) == 1
    hint = merged[0]
    assert np.allclose(hint.edge_vector, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(hint.edge_centroid, np.array([0.5, 0.0, 0.0]))
    assert sorted(hint.face_pair_ids) == [(1, 1), (2, 2), (3, 3)]
    print("test_merge_three_collinear_connector_hints_backward passed.")


def test_compute_connector_hints_on_partition():
    # Create initial mesh
    points, _ = create_dodecahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = mesh.get_trivial_partition()

    # Perforate and split to create two regions
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])
    new_partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    # Compute connector hints
    hints = new_partition.compute_connector_hints(shell_thickness=0.02)

    # Basic checks
    assert isinstance(hints, list)
    assert all(h.region_a != h.region_b for h in hints)
    assert all(h.region_a < h.region_b for h in hints)  # canonicalization
    assert all(np.isclose(np.linalg.norm(h.edge_vector), 1.0) for h in hints)

    # Optional debug output
    for h in hints:
        print(
            f"Connector: {h.region_a} -> {h.region_b}, edge at {h.edge_centroid}, normal A {h.triangle_a_normal}, normal B {h.triangle_b_normal}"
        )

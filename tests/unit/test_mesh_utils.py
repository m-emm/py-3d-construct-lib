import json
import logging
import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from py_3d_construct_lib.connector_hint import ConnectorHint
from py_3d_construct_lib.connector_utils import merge_collinear_connectors
from py_3d_construct_lib.construct_utils import normalize
from py_3d_construct_lib.geometries import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
    create_tetrahedron_geometry,
)
from py_3d_construct_lib.mesh_partition import MeshPartition
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from py_3d_construct_lib.region_edge_feature import RegionEdgeFeature
from py_3d_construct_lib.transformed_region_view import TransformedRegionView

_logger = logging.getLogger(__name__)


# Use only plain pytest-compatible test functions, no unittest framework, no classes, etc.

import os
import struct
import tempfile

from py_3d_construct_lib.mesh_utils import (
    _cross,
    _merge_duplicate_vertices,
    _norm,
    _normalize,
    _sub,
    shell_maps_to_unified_mesh,
    write_shell_maps_to_stl,
    write_stl_binary,
)


def test_vector_operations():
    """Test basic vector operations used in STL export."""
    v1 = (1.0, 2.0, 3.0)
    v2 = (4.0, 5.0, 6.0)

    # Test subtraction
    result = _sub(v1, v2)
    expected = (-3.0, -3.0, -3.0)
    assert result == expected

    # Test cross product
    cross = _cross(v1, v2)
    expected_cross = (
        2.0 * 6.0 - 3.0 * 5.0,
        3.0 * 4.0 - 1.0 * 6.0,
        1.0 * 5.0 - 2.0 * 4.0,
    )
    assert cross == expected_cross

    # Test norm
    norm = _norm(v1)
    expected_norm = (1.0 + 4.0 + 9.0) ** 0.5
    assert abs(norm - expected_norm) < 1e-6

    # Test normalize
    normalized = _normalize(v1)
    normalized_norm = _norm(normalized)
    assert abs(normalized_norm - 1.0) < 1e-6

    # Test zero vector normalization
    zero_normalized = _normalize((0.0, 0.0, 0.0))
    assert zero_normalized == (0.0, 0.0, 0.0)


def test_write_stl_binary_basic():
    """Test basic STL binary writing."""
    # Simple triangle
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    triangles = [(0, 1, 2)]

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_stl_binary(output_path, vertices, triangles, header_text="test triangle")

        # Verify file was created and has correct size
        assert os.path.exists(output_path)

        # Check file structure
        with open(output_path, "rb") as f:
            # Header (80 bytes)
            header = f.read(80)
            assert len(header) == 80
            assert header.startswith(b"test triangle")

            # Triangle count (4 bytes)
            tri_count_bytes = f.read(4)
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            assert tri_count == 1

            # Triangle data (50 bytes per triangle)
            triangle_data = f.read(50)
            assert len(triangle_data) == 50

            # Should be no more data
            remaining = f.read()
            assert len(remaining) == 0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_write_stl_binary_multiple_triangles():
    """Test STL writing with multiple triangles."""
    # Square made of two triangles
    vertices = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]
    triangles = [(0, 1, 2), (0, 2, 3)]  # First triangle  # Second triangle

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_stl_binary(output_path, vertices, triangles)

        # Check triangle count
        with open(output_path, "rb") as f:
            f.seek(80)  # Skip header
            tri_count_bytes = f.read(4)
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            assert tri_count == 2
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_shell_maps_to_unified_mesh_basic():
    """Test basic shell map to unified mesh conversion."""
    # Create simple shell map (like from calculate_materialized_shell_maps)
    simple_shell_maps = {
        0: {
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
            },
            "faces": {0: [0, 1, 2]},
        }
    }

    vertices, triangles = shell_maps_to_unified_mesh(
        simple_shell_maps, remove_inner_faces=False, merge_duplicate_vertices=False
    )

    # Check results
    assert len(vertices) == 3
    assert len(triangles) == 1
    assert triangles[0] == (0, 1, 2)

    # Check vertex values
    expected_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    for i, expected in enumerate(expected_vertices):
        assert vertices[i] == expected


def test_shell_maps_to_unified_mesh_multiple_shells():
    """Test shell map conversion with multiple shells."""
    # Create test shell maps with two triangle prisms
    test_shell_maps = {
        0: {  # First shell (triangle prism)
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),  # inner triangle
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
                3: np.array([0.0, 0.0, 1.0]),  # outer triangle
                4: np.array([1.0, 0.0, 1.0]),
                5: np.array([0.5, 1.0, 1.0]),
            },
            "faces": {
                0: [0, 2, 1],  # bottom (inner)
                1: [3, 4, 5],  # top (outer)
                2: [0, 1, 4],  # side faces
                3: [0, 4, 3],
            },
        },
        1: {  # Second shell (adjacent triangle prism)
            "vertexes": {
                0: np.array([1.0, 0.0, 0.0]),  # inner triangle (shared edge)
                1: np.array([2.0, 0.0, 0.0]),
                2: np.array([1.5, 1.0, 0.0]),
                3: np.array([1.0, 0.0, 1.0]),  # outer triangle
                4: np.array([2.0, 0.0, 1.0]),
                5: np.array([1.5, 1.0, 1.0]),
            },
            "faces": {
                0: [0, 2, 1],  # bottom (inner)
                1: [3, 4, 5],  # top (outer)
                2: [0, 1, 4],  # side faces
                3: [0, 4, 3],
            },
        },
    }

    vertices, triangles = shell_maps_to_unified_mesh(
        test_shell_maps, remove_inner_faces=False, merge_duplicate_vertices=False
    )

    # Should have vertices from both shells
    assert len(vertices) == 12  # 6 vertices per shell * 2 shells
    assert len(triangles) == 8  # 4 faces per shell * 2 shells


def test_merge_duplicate_vertices():
    """Test vertex merging functionality."""
    # Create vertices with duplicates
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),  # Duplicate of vertex 0
        (1.0, 0.0, 0.0001),  # Near duplicate of vertex 1
    ]
    triangles = [(0, 1, 2), (1, 2, 3)]

    merged_vertices, merged_triangles = _merge_duplicate_vertices(
        vertices, triangles, tolerance=1e-3
    )

    # Should have fewer vertices after merging
    assert len(merged_vertices) < len(vertices)

    # All triangle indices should be valid
    for triangle in merged_triangles:
        for vertex_idx in triangle:
            assert vertex_idx < len(merged_vertices)


def test_merge_duplicate_vertices_removes_degenerate_triangles():
    """Test that degenerate triangles are removed during vertex merging."""
    vertices = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0001),  # Very close to vertex 0
        (1.0, 0.0, 0.0),
    ]
    triangles = [(0, 1, 2)]  # Will become degenerate after merging

    merged_vertices, merged_triangles = _merge_duplicate_vertices(
        vertices, triangles, tolerance=1e-3
    )

    # Degenerate triangle should be removed
    assert len(merged_triangles) == 0


def test_write_shell_maps_to_stl_integration():
    """Test complete integration from shell maps to STL file."""
    # Create simple test shell map
    test_shell_maps = {
        0: {
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
                3: np.array([0.0, 0.0, 1.0]),
                4: np.array([1.0, 0.0, 1.0]),
                5: np.array([0.5, 1.0, 1.0]),
            },
            "faces": {
                0: [0, 2, 1],  # bottom
                1: [3, 4, 5],  # top
                2: [0, 1, 4],  # side
                3: [0, 4, 3],  # side
            },
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_shell_maps_to_stl(
            output_path,
            test_shell_maps,
            header_text="test shell mesh",
            remove_inner_faces=True,
            merge_duplicate_vertices=True,
        )

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify it's a valid STL file
        with open(output_path, "rb") as f:
            # Check header
            header = f.read(80)
            assert len(header) == 80

            # Check triangle count is reasonable
            tri_count_bytes = f.read(4)
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            assert tri_count > 0
            assert tri_count < 100  # Reasonable upper bound for our test data
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_numpy_array_conversion():
    """Test that numpy arrays are properly converted to tuples."""
    shell_maps_with_numpy = {
        0: {
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
            },
            "faces": {0: [0, 1, 2]},
        }
    }

    vertices, triangles = shell_maps_to_unified_mesh(shell_maps_with_numpy)

    # All vertices should be tuples, not numpy arrays
    for vertex in vertices:
        assert isinstance(vertex, tuple)
        assert len(vertex) == 3
        for coord in vertex:
            assert isinstance(coord, float)


def test_empty_shell_maps():
    """Test handling of empty shell maps."""
    empty_shell_maps = {}

    vertices, triangles = shell_maps_to_unified_mesh(empty_shell_maps)

    assert len(vertices) == 0
    assert len(triangles) == 0


def test_compute_normals_false():
    """Test STL writing with compute_normals=False."""
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    triangles = [(0, 1, 2)]

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_stl_binary(output_path, vertices, triangles, compute_normals=False)

        # Verify file was created
        assert os.path.exists(output_path)

        # Check that normals are zero
        with open(output_path, "rb") as f:
            f.seek(84)  # Skip header + triangle count
            normal_bytes = f.read(12)  # 3 floats for normal
            normal = struct.unpack("<3f", normal_bytes)
            assert normal == (0.0, 0.0, 0.0)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_first_function():
    assert True

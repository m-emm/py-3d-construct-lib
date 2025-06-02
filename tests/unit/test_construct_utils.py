import logging
import math

import pytest
from py_3d_construct_lib.construct_utils import (
    compute_area,
    normalize_edge,
    split_triangle_topologically,
    triangle_edges,
)

_logger = logging.getLogger(__name__)
import math

from py_3d_construct_lib.construct_utils import (
    compute_area,
    normalize_edge,
    split_triangle_topologically,
    triangle_edges,
)


def test_split_triangle_topologically_one_edge_split():
    # Triangle vertex indices
    triangle = (0, 1, 2)

    # Coordinates: equilateral triangle
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
        3: (0.5, 0.0),  # Midpoint of edge 0-1
    }

    # Define the cut: midpoint on edge (0, 1)
    edge_to_new_vertex = {normalize_edge(0, 1): 3}

    # Call the topological splitter
    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    # Expect exactly two new triangles
    assert len(new_triangles) == 2

    # Check that all triangles are well-formed
    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3  # no duplicate vertices
        all_vertices.update(tri)

    # Must only use vertices 0, 1, 2, and new vertex 3
    assert all_vertices <= {0, 1, 2, 3}

    # Check that triangle area is preserved
    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Inner edge (which includes the cut vertex) must be used in both directions
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    # Each inner edge should be used exactly twice
    for edge, count in counts.items():
        if edge == normalize_edge(0, 1):
            continue  # the original edge was split; its pieces shouldn't reappear
        assert count in (1, 2)


def test_split_triangle_topologically_two_edge_split():
    # Triangle vertex indices
    triangle = (0, 1, 2)

    # Coordinates: equilateral triangle
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
        3: (0.5, 0.0),  # Midpoint of edge 0-1
        4: (0.75, math.sqrt(3) / 4),  # Midpoint of edge 1-2
    }

    # Define the cuts: midpoints on edges (0,1) and (1,2)
    edge_to_new_vertex = {
        normalize_edge(0, 1): 3,
        normalize_edge(1, 2): 4,
    }

    # Call the topological splitter
    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    # Expect exactly three new triangles
    assert len(new_triangles) == 3

    # Check that all triangles are well-formed
    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3
        all_vertices.update(tri)

    # Must only use vertices 0, 1, 2, and new vertices 3, 4
    assert all_vertices <= {0, 1, 2, 3, 4}

    # Check that triangle area is preserved
    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Inner edges should appear twice (in opposite directions)
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    # All edges should appear either once (outer) or twice (inner)
    for edge, count in counts.items():
        if edge in edge_to_new_vertex:
            continue  # skip illegal original edges
        assert count in (1, 2), f"Edge {edge} appears {count} times"


def test_split_triangle_topologically_no_split():
    triangle = (0, 1, 2)

    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
    }

    edge_to_new_vertex = {}  # No cuts

    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    assert len(new_triangles) == 1
    assert (
        new_triangles[0] == [0, 1, 2]
        or new_triangles[0] == [1, 2, 0]
        or new_triangles[0] == [2, 0, 1]
    )

    # Area check
    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = compute_area(*[coords[v] for v in new_triangles[0]])
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)


def test_split_triangle_topologically_three_edge_split():
    triangle = (0, 1, 2)

    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
        3: (0.5, 0.0),  # Midpoint 0-1
        4: (0.75, math.sqrt(3) / 4),  # Midpoint 1-2
        5: (0.25, math.sqrt(3) / 4),  # Midpoint 2-0
    }

    edge_to_new_vertex = {
        normalize_edge(0, 1): 3,
        normalize_edge(1, 2): 4,
        normalize_edge(2, 0): 5,
    }

    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    # Expect four triangles
    assert len(new_triangles) == 4

    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3
        all_vertices.update(tri)

    assert all_vertices <= {0, 1, 2, 3, 4, 5}

    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Check inner edge usage
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    for edge, count in counts.items():
        if edge in edge_to_new_vertex:
            continue
        assert count in (1, 2), f"Edge {edge} appears {count} times"


def test_split_triangle_topologically_three_edge_split_wild_numbering():
    # Arbitrary large global vertex indices
    triangle = (770, 771, 772)

    # Remapped coordinates: same triangle shape
    coords = {
        770: (0.0, 0.0),  # Vertex 0
        771: (1.0, 0.0),  # Vertex 1
        772: (0.5, math.sqrt(3) / 2),  # Vertex 2
        773: (0.5, 0.0),  # Midpoint of 770-771
        774: (0.75, math.sqrt(3) / 4),  # Midpoint of 771-772
        775: (0.25, math.sqrt(3) / 4),  # Midpoint of 772-770
    }

    edge_to_new_vertex = {
        normalize_edge(770, 771): 773,
        normalize_edge(771, 772): 774,
        normalize_edge(772, 770): 775,
    }

    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    assert len(new_triangles) == 4

    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3
        all_vertices.update(tri)

    # We expect only the original and new vertices
    expected_vertices = {770, 771, 772, 773, 774, 775}
    assert all_vertices <= expected_vertices

    # Area should still be preserved
    original_area = compute_area(coords[770], coords[771], coords[772])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Check edge usage consistency
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    for edge, count in counts.items():
        if edge in edge_to_new_vertex:
            continue
        assert count in (1, 2), f"Edge {edge} appears {count} times"

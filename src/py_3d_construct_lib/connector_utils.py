import logging
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
from py_3d_construct_lib.connector_hint import ConnectorHint
from py_3d_construct_lib.construct_utils import compute_triangle_normal, normalize

_logger = logging.getLogger(__name__)


def compute_connector_hints_from_shell_maps(
    mesh_faces: np.ndarray,
    face_to_region: dict[int, int],
    shell_maps: dict[int, dict],
    vertex_index_map: dict[int, dict],
) -> List[ConnectorHint]:
    edge_to_faces = defaultdict(list)

    # 1. Build edge -> [(face_index, region)] mapping
    for f_idx, region in face_to_region.items():
        face = mesh_faces[f_idx]
        for i in range(3):
            a, b = sorted((face[i], face[(i + 1) % 3]))
            edge_to_faces[(a, b)].append((f_idx, region))

    connector_hints = []

    for edge, face_region_pairs in edge_to_faces.items():
        if len(face_region_pairs) != 2:
            raise ValueError(f"The edge {edge} is not shared by exactly two faces.")

        (f_a, r_a), (f_b, r_b) = face_region_pairs
        if r_a == r_b:
            continue

        if r_a > r_b:
            (f_a, r_a), (f_b, r_b) = (f_b, r_b), (f_a, r_a)

        face_a = mesh_faces[f_a]
        shared_indices = set(face_a) & set(mesh_faces[f_b])

        # preserve winding from face_a
        ordered_shared = []
        for i in range(3):
            a, b = face_a[i], face_a[(i + 1) % 3]
            if a in shared_indices and b in shared_indices:
                ordered_shared = [a, b]
                break

        if len(ordered_shared) != 2:
            raise ValueError(f"Could not find shared edge between {f_a} and {f_b}")

        vi1, vi2 = ordered_shared

        def get_vertex(face_id, vi):
            vi_local = vertex_index_map[face_id]["inner"][vi]
            return shell_maps[face_id]["vertexes"][vi_local]

        p1 = get_vertex(f_a, vi1)
        p2 = get_vertex(f_a, vi2)

        edge_vec = normalize(p2 - p1)
        edge_mid = (p1 + p2) / 2

        tri_a = [get_vertex(f_a, vi) for vi in mesh_faces[f_a]]
        tri_b = [get_vertex(f_b, vi) for vi in mesh_faces[f_b]]

        n_a = normalize(compute_triangle_normal(*tri_a))
        n_b = normalize(compute_triangle_normal(*tri_b))

        hint = ConnectorHint(
            region_a=r_a,
            region_b=r_b,
            triangle_a_vertices=tuple(tri_a),
            triangle_b_vertices=tuple(tri_b),
            triangle_a_normal=n_a,
            triangle_b_normal=n_b,
            edge_vector=edge_vec,
            edge_centroid=edge_mid,
            start_vertex=p1,
            end_vertex=p2,
        )
        connector_hints.append(hint)

    return connector_hints


def transform_connector_hint(
    hint: ConnectorHint, transform: np.ndarray
) -> ConnectorHint:
    """
    Apply an affine transformation to a ConnectorHint.
    Positions (vertices, centroids) are fully transformed.
    Vectors (normals, edge vectors) are only rotated.

    Parameters:
    -----------
    hint : ConnectorHint
        The original connector hint.
    transform : np.ndarray
        A 4x4 affine transformation matrix.

    Returns:
    --------
    ConnectorHint
        A new connector hint with transformed data.
    """

    def transform_point(p):
        p_h = np.append(p, 1.0)
        return (transform @ p_h)[:3]

    def rotate_vector(v):
        R = transform[:3, :3]
        return R @ v

    return ConnectorHint(
        region_a=hint.region_a,
        region_b=hint.region_b,
        triangle_a_vertices=tuple(transform_point(v) for v in hint.triangle_a_vertices),
        triangle_b_vertices=tuple(transform_point(v) for v in hint.triangle_b_vertices),
        triangle_a_normal=rotate_vector(hint.triangle_a_normal),
        triangle_b_normal=rotate_vector(hint.triangle_b_normal),
        edge_vector=rotate_vector(hint.edge_vector),
        edge_centroid=transform_point(hint.edge_centroid),
        start_vertex=transform_point(hint.start_vertex),
        end_vertex=transform_point(hint.end_vertex),
        original_edges=list(hint.original_edges),
        face_pair_ids=list(hint.face_pair_ids),
    )

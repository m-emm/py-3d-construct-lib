from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class ConnectorHint:
    region_a: int
    region_b: int
    triangle_a_vertices: Tuple[np.ndarray, np.ndarray, np.ndarray]
    triangle_b_vertices: Tuple[np.ndarray, np.ndarray, np.ndarray]
    triangle_a_normal: np.ndarray
    triangle_b_normal: np.ndarray
    edge_vector: np.ndarray
    edge_centroid: np.ndarray
    start_vertex: np.ndarray
    end_vertex: np.ndarray
    # Optional fields to keep track of merged provenance
    original_edges: list[Tuple[int, int]] = field(default_factory=list)
    face_pair_ids: list[Tuple[int, int]] = field(default_factory=list)

    def __repr__(self):
        retval = "\n".join(
            [
                "\nConnectorHint(",
                f"  region_a={self.region_a}, region_b={self.region_b}",
                f"  edge_centroid={self.edge_centroid}",
                f"  edge_vector={self.edge_vector}",
                f"  start_vertex={self.start_vertex}",
                f"  end_vertex={self.end_vertex}",
                f"  triangle_a_vertices={self.triangle_a_vertices}",
                f"  triangle_b_vertices={self.triangle_b_vertices}",
                f"  triangle_a_normal={self.triangle_a_normal}",
                f"  triangle_b_normal={self.triangle_b_normal}",
                f"  original_edges={self.original_edges}",
                f"  face_pair_ids={self.face_pair_ids}",
            ]
        )
        return retval + ")"

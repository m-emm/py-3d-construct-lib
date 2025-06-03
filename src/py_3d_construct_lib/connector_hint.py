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
    # Optional fields to keep track of merged provenance
    original_edges: list[Tuple[int, int]] = field(default_factory=list)
    face_pair_ids: list[Tuple[int, int]] = field(default_factory=list)

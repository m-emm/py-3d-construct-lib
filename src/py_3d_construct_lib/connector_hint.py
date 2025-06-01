from dataclasses import dataclass

import numpy as np


@dataclass
class ConnectorHint:
    region_a: int
    region_b: int

    triangle_a_vertices: tuple[np.ndarray, np.ndarray, np.ndarray]
    triangle_b_vertices: tuple[np.ndarray, np.ndarray, np.ndarray]

    triangle_a_normal: np.ndarray
    triangle_b_normal: np.ndarray

    edge_vector: np.ndarray
    edge_centroid: np.ndarray

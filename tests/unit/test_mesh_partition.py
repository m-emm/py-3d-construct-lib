import logging
from collections import defaultdict

import numpy as np
from py_3d_construct_lib.construct_utils import create_dodecahedron_geometry
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from py_3d_construct_lib.transformed_region_view import TransformedRegionView

_logger = logging.getLogger(__name__)


def test_perforated():

    points, _ = create_dodecahedron_geometry(1.0)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = mesh.get_trivial_partition()
    partition_perforated = partition.perforated(
        np.array([0, 0, 0]), np.array([0, 0, 1])
    )

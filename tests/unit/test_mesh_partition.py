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

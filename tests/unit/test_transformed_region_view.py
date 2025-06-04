import numpy as np
from py_3d_construct_lib.connector_hint import ConnectorHint
from py_3d_construct_lib.connector_utils import compute_connector_hints_from_shell_maps
from py_3d_construct_lib.construct_utils import create_dodecahedron_geometry, normalize
from py_3d_construct_lib.face_point_cloud import sphere_radius
from py_3d_construct_lib.geometries import (
    create_cube_geometry,
    create_tetrahedron_geometry,
)
from py_3d_construct_lib.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from py_3d_construct_lib.spherical_tools import (
    cartesian_to_spherical_jackson,
    spherical_to_cartesian_jackson,
)
from py_3d_construct_lib.transformed_region_view import TransformedRegionView


def test_transformed_shell_map():
    # Step 1: Create geometry (a cube approximated as a sphere)
    points, _ = create_cube_geometry(sphere_radius)

    # Step 2: Create and partition the mesh
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = mesh.get_trivial_partition()

    # Step 3: Perforate and split into two regions
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 0, 1])
    )

    # Step 4: Get a region view and apply a transformation (e.g., small translation)
    region_view = partition.region_view(0).translated(1.0, 0.0, 0.0)

    # Step 5: Compute transformed shell maps
    shell_maps, vertex_index_map = region_view.get_transformed_materialized_shell_maps(
        shell_thickness=0.1
    )

    # Basic checks
    assert isinstance(shell_maps, dict)
    assert isinstance(vertex_index_map, dict)
    assert len(shell_maps) > 0

    for face_id, shell_map in shell_maps.items():
        assert "vertexes" in shell_map
        assert "faces" in shell_map
        verts = shell_map["vertexes"]
        faces = shell_map["faces"]
        assert isinstance(verts, dict)
        assert isinstance(faces, dict)
        for v in verts.values():
            assert v.shape == (3,)
            assert v[0] > 0.5  # confirm that translation x+1.0 took effect

    for face_id, vmap in vertex_index_map.items():
        assert "inner" in vmap and "outer" in vmap
        assert len(vmap["inner"]) == 3
        assert len(vmap["outer"]) == 3


def test_compute_connector_hints_on_transformed_region_view():
    # Step 1: Generate icosahedron geometry
    points, _ = create_cube_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = mesh.get_trivial_partition()

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 0, 1])
    )

    region_view = partition.region_view(0)

    num_faces = region_view.num_faces()
    print(f"Region {region_view.region_id} has {num_faces} faces")

    # Compute connector hints
    hints = region_view.compute_transformed_connector_hints(shell_thickness=0.02)

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


def test_compute_connector_hints_merge_tetrahedron():

    sphere_radius = 30
    shell_thickness = sphere_radius * 0.05

    shrink_border = 0.3

    # Step 1: Generate geometry
    points, _ = create_tetrahedron_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = mesh.get_trivial_partition()

    partition = partition.perforate_and_split_region_by_plane(
        0, plane_point=np.array([0, 0, 0]), plane_normal=np.array([0, 1, 1])
    )

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    region_views = []

    for region_id in partition.get_regions():
        view = partition.region_view(region_id)

        if region_id == 0:
            view = view.rotated(np.deg2rad(180), axis=(1, 0, 0))

        else:
            view = view.rotated(np.deg2rad(0), axis=(1, 0, 0))
        region_views.append(view)

    # Step 5: Fuse solids per region
    parts = {}
    for region_view in region_views:
        region_id = region_view.region_id

        connector_hints = region_view.compute_transformed_connector_hints(
            shell_thickness, merge_connectors=False
        )

        edge_vectors_int = [
            tuple([int(q) for q in 1000 * np.round(h.edge_vector, 3)])
            for h in connector_hints
        ]
        unique_edge_vectors = set(edge_vectors_int)
        print(f"Unique edge vectors: {unique_edge_vectors}")

        assert len(unique_edge_vectors) == len(
            connector_hints
        ), f"Duplicate edge vectors found: {len(unique_edge_vectors)} unique vs {len(connector_hints)} total"

        print(f"connector_hints: \n{connector_hints}")

        connector_hints_merged = region_view.compute_transformed_connector_hints(
            shell_thickness, merge_connectors=True
        )

        assert len(connector_hints) == len(connector_hints_merged)


def test_lay_flat_optimal():
    # Step 1: Generate geometry
    points, _ = create_tetrahedron_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = mesh.get_trivial_partition()

    partition = partition.perforate_and_split_region_by_plane(
        0, plane_point=np.array([0, 0, 0]), plane_normal=np.array([0, 1, 1])
    )

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    region_views = []

    for region_id in partition.get_regions():
        view = partition.region_view(region_id)
        region_views.append(view)

    # Step 5: Fuse solids per region
    tol = 1e-5  # tolerance for printability score
    for region_view in region_views:
        region_view = region_view.lay_flat_optimally_printable()

        assert region_view.printability_score() >= 0.5 - tol


def test_lay_flat_on_edge():
    print("Generating base icosahedron...")

    # Step 1: Generate icosahedron geometry
    points, _ = create_dodecahedron_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = mesh.get_trivial_partition()

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
    )

    partition = partition.perforate_and_split_region_by_plane(
        region_id=1,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 1, 0]),
    )

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")
    region_views = []

    for region_id in partition.get_regions():
        view = partition.region_view(region_id)
        region_views.append(view)

    region_view = region_views[1]

    region_view = region_view.lay_flat_on_boundary_edges_for_printability()

    assert (
        region_view.printability_score() > 0.1
    ), "Printability score should be at least 0.1"


def test_numerical_instability():
    sphere_radius = 30
    shell_thickness = sphere_radius * 0.05
    shrink_border = 0

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=30, radius=sphere_radius
    )

    for i, v in enumerate(mesh.vertices):
        print(f"Vertex: {i}: {v}")

    new_vertices = []

    random_size = sphere_radius * 0.1
    new_vertices = []

    for v in mesh.vertices:
        r, theta, phi = cartesian_to_spherical_jackson(v)
        # r += np.random.uniform(-random_size, random_size)

        print(
            f"Vertex: Cartesian: {v} -> Spherical: (r={r:.2f}, theta={theta:.2f}, phi={phi:.2f})"
        )

        back = spherical_to_cartesian_jackson((r, theta, phi))
        # back = [ 0.0 if abs(coord) < 1e-6 else coord for coord in back ]

        new_vertices.append([back[0], back[1], back[2]])

    new_vertices = np.array(new_vertices)

    # for i, v in enumerate(mesh.vertices):
    #     assert np.allclose(v, new_vertices[i])

    assert len(new_vertices) == len(mesh.vertices)

    for i, v in enumerate(new_vertices):
        print(f"Vertex: {i}: {v}")

    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(new_vertices)

    for f in sorted([tuple(f) for f in mesh.faces]):
        print(f"Mesh face: {f}")

    partition = mesh.get_trivial_partition()

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
    )

    shell_maps, _ = partition.mesh.calculate_materialized_shell_maps(
        shell_thickness=shell_thickness,
        shrinkage=0,
        shrink_border=shrink_border,
    )

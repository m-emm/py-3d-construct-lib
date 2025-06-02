import copy
import logging
from collections import Counter, defaultdict, deque

import numpy as np
from py_3d_construct_lib.construct_utils import (
    compute_barycentric_coords,
    normalize_edge,
    triangle_edges,
)
from py_3d_construct_lib.mesh_partition import MeshPartition
from py_3d_construct_lib.spherical_tools import (
    cartesian_to_spherical_jackson,
    spherical_to_cartesian_jackson,
)
from scipy.spatial import ConvexHull

_logger = logging.getLogger(__name__)


def calc_edge_to_triangle_map(triangles):
    edge_to_tri = defaultdict(list)

    for i, tri in enumerate(triangles):
        for edge in triangle_edges(tri):
            edge_to_tri[normalize_edge(*edge)].append(i)

    return edge_to_tri


def propagate_consistent_winding(triangles):
    triangles = [list(tri) for tri in triangles]
    edge_to_tri = calc_edge_to_triangle_map(triangles)

    visited = set()
    queue = deque([0])  # start with triangle 0
    visited.add(0)

    while queue:
        current = queue.popleft()
        current_tri = triangles[current]

        for edge in triangle_edges(current_tri):
            edge_key = normalize_edge(*edge)
            neighbors = edge_to_tri[edge_key]

            for neighbor in neighbors:
                if neighbor == current or neighbor in visited:
                    continue

                neighbor_tri = triangles[neighbor]

                # Check how this edge appears in neighbor
                if edge in triangle_edges(neighbor_tri):
                    # Same direction: flip neighbor
                    triangles[neighbor] = neighbor_tri[::-1]

                # Mark visited and continue
                visited.add(neighbor)
                queue.append(neighbor)

    return np.array(triangles)


def is_valid_path(vertex_path, edge_graph):
    return all(
        edge_graph.has_edge(vertex_path[i], vertex_path[i + 1])
        for i in range(len(vertex_path) - 1)
    )


def walk_length(vertex_path, edge_graph):
    return sum(
        edge_graph[vertex_path[i]][vertex_path[i + 1]]["weight"]
        for i in range(len(vertex_path) - 1)
    )


def shrink_triangle(A, B, C, border_width, epsilon=1e-6):
    def compute_offset_point(p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < epsilon or norm2 < epsilon:
            raise ValueError("Degenerate triangle corner with zero-length edge.")

        v1n = v1 / norm1
        v2n = v2 / norm2

        dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
        angle = np.arccos(dot)
        sin_half_angle = np.sin(angle / 2)

        if sin_half_angle < epsilon:
            raise ValueError(
                f"Corner too sharp (angle={np.degrees(angle):.2f}°), cannot safely shrink."
            )

        offset_length = border_width / sin_half_angle

        # Require offset to be smaller than both adjacent edge lengths
        if offset_length > min(norm1, norm2):
            raise ValueError(
                f"Offset {offset_length:.4f} too large for triangle at corner with edge lengths "
                f"{norm1:.4f} and {norm2:.4f}."
            )

        bisector = v1n + v2n
        bisector /= np.linalg.norm(bisector)
        return p0 + bisector * offset_length

    A_new = compute_offset_point(A, B, C)
    B_new = compute_offset_point(B, C, A)
    C_new = compute_offset_point(C, A, B)

    return A_new, B_new, C_new


class PartitionableSpheroidTriangleMesh:
    def __init__(self, vertices, faces, vertex_labels=None):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)

        for face in self.faces:
            assert len(face) == 3, "All faces must be triangles"

        # check if all edges appeear twice, in both directions

        edges_by_canoncial_edge = defaultdict(list)
        for i, face in enumerate(self.faces):
            for j in range(3):
                a, b = (face[j], face[(j + 1) % 3])
                edges_by_canoncial_edge[tuple(sorted((a, b)))].append((a, b))

        for canonical_edge, edges in edges_by_canoncial_edge.items():
            if len(edges) != 2:
                raise ValueError(
                    f"Edge {canonical_edge} appears {len(edges)} times, expected 2"
                )
            assert edges[0] == (
                edges[1][1],
                edges[1][0],
            ), f"Edge {canonical_edge} appears twice in same order"

        if vertex_labels is None:
            self.vertex_labels = [str(i) for i in range(len(self.vertices))]
        else:

            self.vertex_labels = vertex_labels
            assert len(self.vertex_labels) == len(
                self.vertices
            ), "Vertex labels must match the number of vertices"
            assert isinstance(self.vertex_labels, list), "Vertex labels must be a list"
            assert all(
                isinstance(label, str) for label in self.vertex_labels
            ), "All vertex labels must be strings"

    def get_vertex_triangles(self, vertex_index):
        """
        Returns a list of triangle indices that contain the given vertex.
        """
        triangles = []
        for i, face in enumerate(self.faces):
            if vertex_index in face:
                triangles.append(i)
        return triangles

    def get_vertices_by_label(self, label):
        """
        Returns a list of vertex indices that have the given label.
        """

        return [i for i, v in enumerate(self.vertex_labels) if v == label]

    def triangle_area(self, triangle_vertex_indices):

        a, b, c = [self.vertices[i] for i in triangle_vertex_indices]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    def triangle_centroid(self, triangle_vertex_indices):
        a, b, c = [self.vertices[i] for i in triangle_vertex_indices]
        return (a + b + c) / 3.0

    def total_area(self):
        return sum(self.triangle_area(face) for face in self.faces)

    def calculate_materialized_shell_maps(
        self, shell_thickness, shrinkage=0, shrink_border=0
    ):
        """
        Calculate the materialized shell maps for the mesh.
        This will create a triangle prism for each triangle in the mesh, as a traditional face vertex map for each.

        """

        shell_maps = {}

        sphere_center = np.mean(self.vertices, axis=0)
        spherical_vertexes = [
            cartesian_to_spherical_jackson(v - sphere_center) for v in self.vertices
        ]

        for i, face in enumerate(self.faces):

            # get the vertexes of the triangle
            triangle_spherical_vertexes = [
                spherical_vertexes[face[0]],
                spherical_vertexes[face[1]],
                spherical_vertexes[face[2]],
            ]

            # create the shell triangle geometry
            maps = self.create_shell_triangle_geometry(
                triangle_spherical_vertexes,
                sphere_center=sphere_center,
                shell_thickness=shell_thickness,
                shrinkage=shrinkage,
                shrink_border=shrink_border,
            )

            # add the maps to the shell_maps
            shell_maps[i] = {
                "vertexes": maps["vertexes"],
                "faces": maps["faces"],
            }

        return shell_maps

    def get_traditional_face_vertex_maps(self):
        """
        Returns a traditional face vertex map for the mesh.
        """
        maps = {
            "vertexes": {i: v for i, v in enumerate(self.vertices)},
            "faces": {i: tuple(face) for i, face in enumerate(self.faces)},
        }
        return maps

    def get_projected_inner_triangle_vertices(
        self, face_index: int, shell_thickness: float
    ) -> list[np.ndarray]:
        """
        Returns the 3 inner (projected) triangle vertices for a given face index and shell thickness.
        """
        face = self.faces[face_index]
        spherical_coords = [
            cartesian_to_spherical_jackson(self.vertices[i]) for i in face
        ]
        sphere_center = np.mean(self.vertices, axis=0)

        return self._project_inner_triangle(
            spherical_coords, shell_thickness, sphere_center
        )

    @staticmethod
    def _project_inner_triangle(
        spherical_vertexes, shell_thickness: float, sphere_center: np.ndarray
    ):
        """
        Given spherical triangle vertices, projects them inward onto a parallel plane using ray-plane intersection.
        This reproduces the inner triangle geometry for a shell.
        """
        assert len(spherical_vertexes) == 3

        outer_verts = [
            spherical_to_cartesian_jackson(
                v, radius_offset=0, sphere_center=sphere_center
            )
            for v in spherical_vertexes
        ]
        v0, v1, v2 = outer_verts

        tri_normal = np.cross(v1 - v0, v2 - v0)
        tri_normal /= np.linalg.norm(tri_normal)

        plane_point = v0 + (-shell_thickness) * tri_normal

        def intersect_ray_plane(ray_origin, ray_dir, plane_point, plane_normal):
            denom = np.dot(ray_dir, plane_normal)
            if abs(denom) < 1e-8:
                raise ValueError("Ray is parallel to plane")
            t = np.dot(plane_point - ray_origin, plane_normal) / denom
            return ray_origin + t * ray_dir

        inner_verts = []
        for v in outer_verts:
            ray_dir = v - sphere_center
            ray_dir /= np.linalg.norm(ray_dir)
            inner = intersect_ray_plane(sphere_center, ray_dir, plane_point, tri_normal)
            inner_verts.append(inner)

        return inner_verts

    @staticmethod
    def create_shell_triangle_geometry(
        triangle_spherical_vertexes,
        sphere_center,
        shell_thickness,
        shrinkage=0.1,
        shrink_border=0,
    ):
        """
        Improved version: constructs a triangle prism where the inner triangle
        lies on a plane parallel to the outer triangle, offset by shell_thickness,
        but vertices are projected radially from the sphere center.
        """

        if len(triangle_spherical_vertexes) != 3:
            raise ValueError("triangle_spherical_vertexes must have 3 elements")

        for i in range(3):
            if len(triangle_spherical_vertexes[i]) != 3:
                raise ValueError("Each vertex must be (r, theta, phi)")

        outer_verts = [
            spherical_to_cartesian_jackson(
                v, radius_offset=0, sphere_center=sphere_center
            )
            for v in triangle_spherical_vertexes
        ]

        inner_verts = PartitionableSpheroidTriangleMesh._project_inner_triangle(
            triangle_spherical_vertexes, shell_thickness, sphere_center
        )

        all_verts = outer_verts + inner_verts
        centroid = np.mean(all_verts, axis=0)
        for i in range(3):
            outer_verts[i] = outer_verts[i] - shrinkage * (outer_verts[i] - centroid)
            inner_verts[i] = inner_verts[i] - shrinkage * (inner_verts[i] - centroid)

        # Optional: border shrinking
        if shrink_border > 0:
            outer_verts = shrink_triangle(*outer_verts, border_width=shrink_border)
            inner_verts = shrink_triangle(*inner_verts, border_width=shrink_border)

        # Assemble into triangle prism
        vertexes = {i: v for i, v in enumerate(inner_verts)}
        outside_vertexes = {i + 3: v for i, v in enumerate(outer_verts)}
        all_vertices = {**vertexes, **outside_vertexes}

        maps = {
            "vertexes": all_vertices,
            "faces": {
                0: [0, 2, 1],  # bottom
                1: [3, 4, 5],  # top
                2: [0, 1, 4],
                3: [0, 4, 3],
                4: [1, 2, 5],
                5: [1, 5, 4],
                6: [2, 0, 3],
                7: [2, 3, 5],
            },
        }

        return maps

    @classmethod
    def from_traditional_face_vertex_maps(cls, traditional_face_vertex_map):

        vertices = np.array(
            [
                v
                for k, v in sorted(
                    traditional_face_vertex_map["vertexes"].items(),
                    key=lambda item: item[0],
                )
            ]
        )
        faces = np.array(
            [
                f
                for k, f in sorted(
                    traditional_face_vertex_map["faces"].items(),
                    key=lambda item: item[0],
                )
            ]
        )
        return cls(vertices, faces)

    @classmethod
    def from_point_cloud(cls, point_cloud, vertex_labels=None):

        vertices = np.array(point_cloud)

        center = np.mean(vertices, axis=0)

        centered_vertices = vertices - center

        points_r_theta_phi = np.array(
            [cartesian_to_spherical_jackson(p) for p in centered_vertices]
        )

        points_on_unit_sphere_r_theta_phi = np.array(
            [(1, p[1], p[2]) for p in points_r_theta_phi]
        )

        points_for_convex_hull = np.array(
            [
                spherical_to_cartesian_jackson(p)
                for p in points_on_unit_sphere_r_theta_phi
            ]
        )

        hull = ConvexHull(points_for_convex_hull)

        triangles = propagate_consistent_winding(hull.simplices)

        # check if first triangle faces outwards

        triangle_0_normal = np.cross(
            points_for_convex_hull[triangles[0][1]]
            - points_for_convex_hull[triangles[0][0]],
            points_for_convex_hull[triangles[0][2]]
            - points_for_convex_hull[triangles[0][0]],
        )
        triangle_0_normal /= np.linalg.norm(triangle_0_normal)

        triangle_0_centroid = (
            points_for_convex_hull[triangles[0][0]]
            + points_for_convex_hull[triangles[0][1]]
            + points_for_convex_hull[triangles[0][2]]
        ) / 3.0

        if np.dot(triangle_0_normal, triangle_0_centroid) < 0:

            print(f"Flipping triangles to ensure outward normals.")

            # flip all triangles

            triangles = [t[::-1] for t in triangles]

        return cls(vertices, triangles, vertex_labels=vertex_labels)

    @classmethod
    def create_fibonacci_sphere_mesh(cls, num_points, radius=1.0):
        """
        Create a mesh of points on a Fibonacci sphere.
        This is useful for generating evenly distributed points on a sphere.
        """
        phi = np.pi * (3.0 - np.sqrt(5.0))

        points = []
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2
            radius_at_y = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append((x * radius, y * radius, z * radius))
        points = np.array(points)

        return cls.from_point_cloud(points)

    def get_trivial_partition(self):
        """
        Returns a trivial partition of the mesh, where all faces are in the same region.
        This is usefule to then further partition the mesh using the methods of the MeshPartition class.
        """
        face_to_region_map = {i: 0 for i in range(len(self.faces))}
        return MeshPartition(self, face_to_region_map)

    def add_vertex_in_face(self, face_index, barycentric_coords):
        """
        Adds a new vertex inside the specified face using barycentric coordinates.
        Also creates the new faces required to maintain the mesh structure.
        Returns a new mesh object with the new vertex added and the face updated.
        """

        face = self.faces[face_index]
        v0, v1, v2 = [self.vertices[i] for i in face]

        # check if barycentric coordinates are valid
        if len(barycentric_coords) != 3:
            raise ValueError("Barycentric coordinates must have 3 components.")
        if not np.isclose(sum(barycentric_coords), 1.0):
            raise ValueError(
                f"Barycentric coordinates must sum to 1.0, got: {barycentric_coords} (sum={sum(barycentric_coords)})"
            )
        if any(coord < 0 for coord in barycentric_coords):
            raise ValueError(
                f"Barycentric coordinates must be non-negative, got: {barycentric_coords}"
            )
        if len([coord for coord in barycentric_coords if coord < 1e-6]) > 1:
            raise ValueError(
                "Barycentric coordinates must have at most one zero component (i.e., not on a vertex), got: "
                f"{barycentric_coords}"
            )

        new_vertex = (
            barycentric_coords[0] * v0
            + barycentric_coords[1] * v1
            + barycentric_coords[2] * v2
        )

        new_vertices = copy.deepcopy(self.vertices)
        new_faces = copy.deepcopy(self.faces).tolist()
        new_labels = copy.deepcopy(self.vertex_labels)

        new_index = len(self.vertices)
        new_vertices = np.append(new_vertices, [new_vertex], axis=0)

        if any([coord < 1e-6 for coord in barycentric_coords]):
            _logger.info(
                f"add_vertex_in_face: within edge with barycentric coords: {barycentric_coords}, triangle vertex labels: {','.join(self.vertex_labels[i] for i in face)}"
            )
            # On an edge of the triangle
            for i in range(3):
                if barycentric_coords[i] < 1e-6:
                    edge = [face[j] for j in range(3) if j != i]
                    break

            edge = tuple(edge)
            canonical_edge = normalize_edge(*edge)

            edge_to_tri = calc_edge_to_triangle_map(self.faces)
            edge_triangle_indices = edge_to_tri.get(canonical_edge, [])

            if len(edge_triangle_indices) != 2:
                raise ValueError(
                    f"Expected exactly 2 triangles for edge {canonical_edge}, but found {len(edge_triangle_indices)}."
                )

            # remove the two old triangles
            for idx in sorted(edge_triangle_indices, reverse=True):
                new_faces.pop(idx)

            new_labels.append(
                f"{self.vertex_labels[canonical_edge[0]]}__{self.vertex_labels[canonical_edge[1]]}"
            )  # __ is a sign for a new vertex added on an edge

            for tri_index in edge_triangle_indices:
                tri = self.faces[tri_index]
                if edge[0] in tri and edge[1] in tri:
                    a, b = edge
                else:
                    a, b = edge[1], edge[0]  # reverse

                # determine third vertex
                c = next(v for v in tri if v != a and v != b)

                # detect orientation of (a, b, c) in triangle
                ai = list(tri).index(a)
                bi = list(tri).index(b)
                # if they are consecutive in order, (a, b) is the winding
                is_reversed = (bi - ai) % 3 == 2

                if not is_reversed:
                    new_faces.extend(
                        [
                            [a, new_index, c],
                            [new_index, b, c],
                        ]
                    )
                else:
                    new_faces.extend(
                        [
                            [b, new_index, c],
                            [new_index, a, c],
                        ]
                    )
        else:
            _logger.info(
                f"add_vertex_in_face: add inside, baricentric coords: {barycentric_coords}, triangle vertex labels: {','.join(self.vertex_labels[i] for i in face)}"
            )

            # Inside triangle: replace with 3 triangles
            new_labels.append("+".join(self.vertex_labels[i] for i in face))
            new_faces.pop(face_index)
            i0, i1, i2 = face
            new_faces.extend(
                [
                    [i0, i1, new_index],
                    [i1, i2, new_index],
                    [i2, i0, new_index],
                ]
            )

        return PartitionableSpheroidTriangleMesh(new_vertices, new_faces, new_labels)

    def perforate_along_plane(
        self, plane_point: np.ndarray, plane_normal: np.ndarray, epsilon: float = 1e-8
    ):
        return self._perforate_along_plane_bulk(plane_point, plane_normal, epsilon)

    @staticmethod
    def canonicalize_faces(faces: np.ndarray) -> np.ndarray:
        """
        Cycles each triangle (a,b,c) so that the smallest vertex index comes first,
        preserving winding order (i.e. CCW stays CCW).

        Input:
        faces: (N,3) ndarray of triangle vertex indices
        Returns:
        (N,3) ndarray with canonicalized faces
        """
        faces = np.asarray(faces)
        assert (
            faces.ndim == 2 and faces.shape[1] == 3
        ), "Input must be (N,3) triangle array"

        # For each row, compute which index is smallest
        idx_min = np.argmin(faces, axis=1)

        # Rotate each triangle so that min index is first
        canon_faces = np.empty_like(faces)
        for i in range(3):
            canon_faces[idx_min == i] = np.roll(faces[idx_min == i], -i, axis=1)

        return canon_faces

    def _perforate_along_plane_bulk(self, plane_point, plane_normal, epsilon):

        def check_directed_edge_duplicates(faces):

            directed_edges = []

            for f in faces:
                directed_edges.extend(triangle_edges(f))

            directed_edge_by_canonical_edge = defaultdict(list)
            for a, b in directed_edges:
                canonical_edge = normalize_edge(a, b)
                directed_edge_by_canonical_edge[canonical_edge].append((a, b))

            for (
                canonical_edge,
                directed_edges_for_canonical_edge,
            ) in directed_edge_by_canonical_edge.items():
                count = len(directed_edges_for_canonical_edge)
                if count != 2:
                    _logger.info(
                        f"❌ Edge {canonical_edge} appears {count} times, expected 2"
                    )

                unique_edges = set(directed_edges_for_canonical_edge)
                if len(unique_edges) != count:
                    _logger.info(
                        f"❌ Edge {canonical_edge} has non-unique directed edges: {directed_edges_for_canonical_edge}"
                    )

        V_orig = self.vertices
        F_orig = self.faces
        labels_orig = self.vertex_labels
        _logger.info(f"Checking directed edge duplicates in original mesh\n\n")
        check_directed_edge_duplicates(F_orig)

        # Step 1: find every intersected edge, record one new vertex per edge
        edge_to_cutpoint_index = {}
        new_vertices = []
        new_labels = []
        next_index = len(V_orig)

        seen_edges = set()
        for tri in F_orig:
            for k in range(3):
                a = tri[k]
                b = tri[(k + 1) % 3]
                current_edge = normalize_edge(a, b)
                if current_edge in seen_edges:
                    continue
                seen_edges.add(current_edge)

                Va = V_orig[current_edge[0]]
                Vb = V_orig[current_edge[1]]
                d = Vb - Va
                w = Va - plane_point
                denom = np.dot(plane_normal, d)
                if abs(denom) < epsilon:
                    continue

                t = -np.dot(plane_normal, w) / denom
                if 0 < t < 1:
                    ipt = (1 - t) * Va + t * Vb
                    edge_to_cutpoint_index[current_edge] = next_index
                    new_vertices.append(ipt)
                    lbl = f"{labels_orig[current_edge[0]]}__{labels_orig[current_edge[1]]}"
                    new_labels.append(lbl)
                    next_index += 1

        # Build the new vertex‐array and label‐list
        V_new = np.vstack([V_orig, np.array(new_vertices)])
        labels_new = labels_orig + new_labels

        # Step 2: rebuild faces
        F_new = []
        orig_area_sign_cache = {}

        seen_directed_edges = set()

        for tri_index, tri in enumerate(F_orig):

            _logger.info(
                f"\n**************\nProcessing triangle index {tri_index}: {tri}\n***************\n"
            )

            current_triangle_edges = triangle_edges(tri)

            canonical_triangle_edges = [
                normalize_edge(*edge) for edge in current_triangle_edges
            ]

            cuts = []
            for i, edge in enumerate(canonical_triangle_edges):
                if edge in edge_to_cutpoint_index:
                    new_idx = edge_to_cutpoint_index[edge]
                    _logger.info(f"Adding cut at edge {edge} with new index {new_idx}")
                    cuts.append((i, (i + 1) % 3, new_idx))

            k = len(cuts)
            _logger.info(f"\nk={k}\nCuts: {cuts} for triangle {tri}")

            if k == 0:
                # no subdivision
                F_new.append(tri.tolist())

                if any(edge in seen_directed_edges for edge in current_triangle_edges):
                    raise ValueError(
                        "Generated triangle has edges that were already seen, this should not happen."
                    )

                seen_directed_edges.update(current_triangle_edges)

            elif k == 1:
                new_vertices = []
                _logger.info(f"Edge to cutpoint index: {edge_to_cutpoint_index}")
                edges_to_keep = []
                new_edges = []

                for i in range(3):
                    current_edge = (tri[i], tri[(i + 1) % 3])

                    current_canonical_edge = normalize_edge(*current_edge)
                    if current_canonical_edge not in edge_to_cutpoint_index:
                        _logger.info(
                            f"Edge {current_canonical_edge} not in edge_to_cutpoint_index, keeping as is, adding {current_edge} to edges_to_keep"
                        )
                        edges_to_keep.append(current_edge)
                    else:
                        _logger.info(
                            f"Edge {current_canonical_edge} is in edge_to_cutpoint_index"
                        )

                        new_vertex_on_this_edge = edge_to_cutpoint_index[
                            current_canonical_edge
                        ]

                        new_edges.append((current_edge[0], new_vertex_on_this_edge))
                        new_edges.append((new_vertex_on_this_edge, current_edge[1]))
                        new_vertices.append(new_vertex_on_this_edge)

                _logger.info(
                    f"Processing triangle {tri}  in k==1 with cuts: {cuts}, edges(non-canonical): {triangle_edges(tri)}"
                )

                assert len(edges_to_keep) == 2, "Expected exactly two edges to keep"

                shared_vertex = [
                    v
                    for v, count in Counter(
                        [e[0] for e in edges_to_keep] + [e[1] for e in edges_to_keep]
                    ).items()
                    if count > 1
                ][0]
                _logger.info(f"Shared vertex in edges_to_keep: {shared_vertex}")
                new_edge = (shared_vertex, new_vertices[0])
                new_edges.append(new_edge)

                _logger.info(
                    f"Edges to keep: {edges_to_keep}, New edges: {new_edges}, new_vertices: {new_vertices}"
                )

                edges_by_start_vertex = defaultdict(set)
                possible_edges = set(sorted((edges_to_keep + new_edges)))
                for e in possible_edges:
                    edges_by_start_vertex[e[0]].add(e)

                _logger.info(
                    f"Possible edges: {possible_edges}, \nedges_by_start_vertex:\n{edges_by_start_vertex}"
                )

                triangle_0_edges = []
                triangle_0_edges.append(edges_to_keep[0])

                _logger.info(f"Initial triangle_0_edges: {triangle_0_edges}")

                triangle_0_continuation_vertex = triangle_0_edges[0][1]
                triangle_0_edge_candidates = edges_by_start_vertex[
                    triangle_0_continuation_vertex
                ]

                _logger.info(
                    f"Initial triangle_0_edge_candidates: {triangle_0_edge_candidates}, looked for edges starting at vertex {triangle_0_continuation_vertex}"
                )

                triangle_0_edge_candidates = [
                    e for e in triangle_0_edge_candidates if e not in edges_to_keep
                ]

                _logger.info(
                    f"Filtered triangle_0_edge_candidates: {triangle_0_edge_candidates}"
                )

                _logger.info(
                    f"Found {len(triangle_0_edge_candidates)} candidates starting at vertex {edge_to_keep[1]}: {triangle_0_edge_candidates}"
                )
                if len(triangle_0_edge_candidates) != 1:
                    raise ValueError(
                        f"Expected exactly one edge to connect to {edge_to_keep[1]}, found {len(triangle_0_edge_candidates)}"
                    )
                triangle_0_edges.append(next(qq for qq in triangle_0_edge_candidates))
                triangle_0_edges.append(
                    (triangle_0_edges[-1][1], triangle_0_edges[0][0])
                )

                triangle_0_vertex_indices = [edge[0] for edge in triangle_0_edges]
                _logger.info(f"Triangle 0 is {triangle_0_vertex_indices}")

                currently_used_edges = set()
                currently_used_edges.update(
                    [
                        normalize_edge(*edge)
                        for edge in triangle_edges(triangle_0_vertex_indices)
                    ]
                )

                currently_used_non_canonical_edges = set(
                    triangle_edges(triangle_0_vertex_indices)
                )

                currently_used_edges = set(
                    normalize_edge(*edge) for edge in currently_used_non_canonical_edges
                )

                for used_edge in triangle_edges(triangle_0_vertex_indices):
                    # reverse edges now become available
                    possible_edges.add((used_edge[1], used_edge[0]))

                for e in possible_edges:
                    edges_by_start_vertex[e[0]].add(e)

                _logger.info(
                    f"Possible edges after triangle 0: {possible_edges}, currently_used_non_canonical_edges: {currently_used_non_canonical_edges} currently_used_edges    : {currently_used_edges}"
                )

                triangle_1_edges = []

                triangle_1_edges.append(edges_to_keep[1])
                _logger.info(f"Triangle 1 edges so far: {triangle_1_edges}")

                triangle_1_edge_candiates_2 = edges_by_start_vertex[
                    triangle_1_edges[0][1]
                ]
                _logger.info(
                    f"Found {len(triangle_1_edge_candiates_2)} triangle_1_edge_candiates_2 starting at vertex {triangle_1_edges[0][1]}: {triangle_1_edge_candiates_2}"
                )

                triangle_1_edge_candiates_2 = [
                    e
                    for e in triangle_1_edge_candiates_2
                    if e not in currently_used_non_canonical_edges
                ]

                _logger.info(
                    f"Filtered triangle_1_edge_candiates_2: {triangle_1_edge_candiates_2}"
                )

                if len(triangle_1_edge_candiates_2) != 1:
                    raise ValueError(
                        f"Expected exactly one edge to connect to {triangle_1_edges[0][1]}, found {len(triangle_1_edge_candiates_2)}"
                    )
                triangle_1_edges.append(next(qq for qq in triangle_1_edge_candiates_2))

                triangle_1_edges.append(
                    (triangle_1_edges[-1][1], triangle_1_edges[0][0])
                )
                triangle_1_vertex_indices = [edge[0] for edge in triangle_1_edges]

                currently_used_non_canonical_edges.update(
                    triangle_edges(triangle_1_vertex_indices)
                )

                for used_edge in triangle_edges(triangle_1_vertex_indices):
                    # reverse edges now become available
                    possible_edges.add((used_edge[1], used_edge[0]))

                for e in possible_edges:
                    edges_by_start_vertex[e[0]].add(e)

                _logger.info(f"Triangle 1 is {triangle_1_vertex_indices}")

                new_triangles = [
                    triangle_0_vertex_indices,
                    triangle_1_vertex_indices,
                ]

                _logger.info(f"New triangles for {tri}: {new_triangles}")

                all_new_edges_canonical = set()

                for new_tri in new_triangles:
                    new_edges = triangle_edges(new_tri)

                    for edge in new_edges:
                        if edge in seen_directed_edges:
                            raise ValueError(
                                f"Edge {edge} already seen, this should not happen."
                            )
                        normalized_edge = normalize_edge(*edge)

                        all_new_edges_canonical.add(normalized_edge)

                _logger.info(
                    f"All new edges (canonical): {all_new_edges_canonical}, edge_to_cutpoint_index: {edge_to_cutpoint_index}"
                )

                if any(
                    edge in edge_to_cutpoint_index for edge in all_new_edges_canonical
                ):
                    raise ValueError(
                        "Generated triangles contain edges that were already cut, this should not happen."
                    )

                F_new.extend(new_triangles)

            elif k == 2:

                # Triangle indices
                new_vertices = []
                _logger.info(f"Edge to cutpoint index: {edge_to_cutpoint_index}")
                edges_to_keep = []
                new_edges = []

                for i in range(3):

                    current_edge = (tri[i], tri[(i + 1) % 3])

                    current_canonical_edge = normalize_edge(*current_edge)
                    if current_canonical_edge not in edge_to_cutpoint_index:
                        _logger.info(
                            f"Edge {current_canonical_edge} not in edge_to_cutpoint_index, keeping as is, adding {current_edge} to edges_to_keep"
                        )
                        edges_to_keep.append(current_edge)
                    else:
                        _logger.info(
                            f"Edge {current_canonical_edge} is in edge_to_cutpoint_index"
                        )

                        new_vertex_on_this_edge = edge_to_cutpoint_index[
                            current_canonical_edge
                        ]

                        new_edges.append((current_edge[0], new_vertex_on_this_edge))
                        new_edges.append((new_vertex_on_this_edge, current_edge[1]))
                        new_vertices.append(new_vertex_on_this_edge)

                _logger.info(
                    f"Processing triangle {tri} with cuts: {cuts}, edges(non-canonical): {triangle_edges(tri)}"
                )
                _logger.info(
                    f"Edges to keep: {edges_to_keep}, New edges: {new_edges}, new_vertices: {new_vertices}"
                )

                assert len(edges_to_keep) == 1, "Expected exactly one edge to keep"

                edge_to_keep = edges_to_keep[0]

                edges_by_start_vertex = defaultdict(set)
                possible_edges = set(sorted((edges_to_keep + new_edges)))
                for e in possible_edges:
                    edges_by_start_vertex[e[0]].add(e)

                _logger.info(
                    f"Possible edges: {possible_edges}, \nedges_by_start_vertex:\n{edges_by_start_vertex}"
                )

                triangle_0_edges = []
                triangle_0_edges.append(edge_to_keep)
                triangle_0_edge_candidates = edges_by_start_vertex[edge_to_keep[1]]
                _logger.info(
                    f"Found {len(triangle_0_edge_candidates)} candidates starting at vertex {edge_to_keep[1]}: {triangle_0_edge_candidates}"
                )
                if len(triangle_0_edge_candidates) != 1:
                    raise ValueError(
                        f"Expected exactly one edge to connect to {edge_to_keep[1]}, found {len(triangle_0_edge_candidates)}"
                    )
                triangle_0_edges.append(next(qq for qq in triangle_0_edge_candidates))
                triangle_0_edges.append(
                    (triangle_0_edges[-1][1], triangle_0_edges[0][0])
                )

                triangle_0_vertex_indices = [edge[0] for edge in triangle_0_edges]
                _logger.info(f"Triangle 0 is {triangle_0_vertex_indices}")

                currently_used_edges = set()
                currently_used_edges.update(
                    [
                        normalize_edge(*edge)
                        for edge in triangle_edges(triangle_0_vertex_indices)
                    ]
                )

                currently_used_non_canonical_edges = set(
                    triangle_edges(triangle_0_vertex_indices)
                )

                for used_edge in triangle_edges(triangle_0_vertex_indices):
                    # reverse edges now become available
                    possible_edges.add((used_edge[1], used_edge[0]))

                _logger.info(
                    f"currently_used_non_canonical_edges after triangle 0: {currently_used_non_canonical_edges}"
                )
                triangle_1_edges = []
                triangle_1_edge_candidates = edges_by_start_vertex[
                    triangle_0_edges[1][1]
                ]

                _logger.info(
                    f"Found {len(triangle_1_edge_candidates)} candidates starting at vertex {  triangle_0_edges[1][1]}: {triangle_1_edge_candidates}"
                )

                if len(triangle_1_edge_candidates) != 1:
                    raise ValueError(
                        f"Expected exactly one edge to connect to {triangle_0_edges[1][1]}, found {len(triangle_1_edge_candidates)}"
                    )
                triangle_1_edges.append(next(qq for qq in triangle_1_edge_candidates))

                triangle_1_edge_candiates_2 = edges_by_start_vertex[
                    triangle_1_edges[0][1]
                ]
                if len(triangle_1_edge_candiates_2) != 1:
                    raise ValueError(
                        f"Expected exactly one edge to connect to {triangle_1_edges[0][1]}, found {len(triangle_1_edge_candiates_2)}"
                    )
                triangle_1_edges.append(next(qq for qq in triangle_1_edge_candiates_2))

                triangle_1_edges.append(
                    (triangle_1_edges[-1][1], triangle_1_edges[0][0])
                )
                triangle_1_vertex_indices = [edge[0] for edge in triangle_1_edges]

                currently_used_non_canonical_edges.update(
                    triangle_edges(triangle_1_vertex_indices)
                )

                for used_edge in triangle_edges(triangle_1_vertex_indices):
                    # reverse edges now become available
                    possible_edges.add((used_edge[1], used_edge[0]))

                for e in possible_edges:
                    edges_by_start_vertex[e[0]].add(e)

                _logger.info(f"Triangle 1 is {triangle_1_vertex_indices}")

                currently_used_edges.update(
                    [
                        normalize_edge(*edge)
                        for edge in triangle_edges(triangle_1_vertex_indices)
                    ]
                )

                _logger.info(
                    f"currently_used_non_canonical_edges before triangle 2: {currently_used_non_canonical_edges}"
                )

                _logger.info(f"Possible edges before triangle 2: {possible_edges}")
                _logger.info(
                    f"Edges by start vertex before triangle 2: {edges_by_start_vertex}"
                )
                triangle_2_edges = []
                triangle_2_edge_candidates = [
                    e
                    for e in possible_edges
                    if e not in currently_used_non_canonical_edges
                ]

                triangle_2_edge_candidates = [
                    e
                    for e in triangle_2_edge_candidates
                    if e[0] not in tri and e[1] not in tri
                ]

                _logger.info(
                    f"Found {len(triangle_2_edge_candidates)} candidates for triangle 2: {triangle_2_edge_candidates}"
                )

                triangle_2_edges.append(triangle_2_edge_candidates[0])

                triangle_2_edge_candidates_2 = edges_by_start_vertex[
                    triangle_2_edges[0][1]
                ]

                triangle_2_edge_candidates_2_possible = [
                    e
                    for e in triangle_2_edge_candidates_2
                    if normalize_edge(*e) not in currently_used_edges
                ]

                _logger.info(
                    f"Found {len(triangle_2_edge_candidates_2_possible)} really possible candidates starting at vertex {triangle_2_edges[0][1]}: {triangle_2_edge_candidates_2_possible}"
                )
                if len(triangle_2_edge_candidates_2_possible) != 1:
                    raise ValueError(
                        f"Expected exactly one edge to connect to {triangle_2_edges[0][1]}, found {len(triangle_2_edge_candidates_2_possible)}"
                    )
                triangle_2_edges.append(triangle_2_edge_candidates_2_possible[0])
                triangle_2_edges.append(
                    (triangle_2_edges[-1][1], triangle_2_edges[0][0])
                )
                triangle_2_vertex_indices = [edge[0] for edge in triangle_2_edges]

                new_triangles = [
                    triangle_0_vertex_indices,
                    triangle_1_vertex_indices,
                    triangle_2_vertex_indices,
                ]

                _logger.info(f"New triangles for {tri}: {new_triangles}")

                all_new_edges_canonical = set()

                for new_tri in new_triangles:
                    new_edges = triangle_edges(new_tri)

                    for edge in new_edges:
                        if edge in seen_directed_edges:
                            raise ValueError(
                                f"Edge {edge} already seen, this should not happen."
                            )
                        normalized_edge = normalize_edge(*edge)

                        all_new_edges_canonical.add(normalized_edge)

                _logger.info(
                    f"All new edges (canonical): {all_new_edges_canonical}, edge_to_cutpoint_index: {edge_to_cutpoint_index}"
                )

                if any(
                    edge in edge_to_cutpoint_index for edge in all_new_edges_canonical
                ):
                    raise ValueError(
                        "Generated triangles contain edges that were already cut, this should not happen."
                    )

                F_new.extend(new_triangles)

            else:  # k == 3
                # degenerate: plane goes through all three edges.  Copy original.
                F_new.append(tri.tolist())
                if any(edge in seen_directed_edges for edge in current_triangle_edges):
                    raise ValueError(
                        "Generated triangle has edges that were already seen, this should not happen."
                    )
                seen_directed_edges.update(current_triangle_edges)

        F_new = self.canonicalize_faces(F_new)
        check_directed_edge_duplicates(F_new)

        f_new_set = set([tuple(sorted(f)) for f in F_new])
        if len(f_new_set) != len(F_new):
            raise ValueError("Generated faces are not unique, there are duplicates.")
        return PartitionableSpheroidTriangleMesh(V_new, np.array(F_new), labels_new)

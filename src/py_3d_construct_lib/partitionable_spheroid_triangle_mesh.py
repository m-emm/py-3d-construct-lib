import copy
import logging
from collections import Counter, defaultdict, deque

import numpy as np
from py_3d_construct_lib.construct_utils import (
    compute_area,
    compute_barycentric_coords,
    normalize_edge,
    split_triangle_topologically,
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

    def perforate_along_plane(
        self, plane_point, plane_normal, epsilon=1e-8, triangle_indices=None
    ):
        V_orig = self.vertices
        F_orig = self.faces
        labels_orig = self.vertex_labels

        all_tri_indices = range(len(F_orig))
        triangle_indices = set(
            all_tri_indices if triangle_indices is None else triangle_indices
        )

        face_index_mapping = {}

        # Step 1: Find intersected edges and compute new vertices
        edge_to_cutpoint_index = {}
        new_vertices = []
        new_labels = []
        next_index = len(V_orig)
        seen_edges = set()

        for tri_idx in triangle_indices:
            tri = F_orig[tri_idx]
            for a, b in triangle_edges(tri):
                edge = normalize_edge(a, b)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)

                Va, Vb = V_orig[edge[0]], V_orig[edge[1]]
                d = Vb - Va
                w = Va - plane_point
                denom = np.dot(plane_normal, d)

                if abs(denom) < epsilon:
                    continue  # edge is parallel to plane

                t = -np.dot(plane_normal, w) / denom
                if 0 < t < 1:
                    ipt = (1 - t) * Va + t * Vb
                    edge_to_cutpoint_index[edge] = next_index
                    new_vertices.append(ipt)
                    new_labels.append(f"{labels_orig[edge[0]]}__{labels_orig[edge[1]]}")
                    next_index += 1

        # --- Step 1.5: Expand triangle_indices to include all triangles that touch cut edges
        if edge_to_cutpoint_index:
            edge_to_tri_indices = defaultdict(set)
            for tri_idx, tri in enumerate(F_orig):
                for edge in triangle_edges(tri):
                    norm_edge = normalize_edge(*edge)
                    edge_to_tri_indices[norm_edge].add(tri_idx)

            affected_tri_indices = set()
            for cut_edge in edge_to_cutpoint_index:
                affected_tri_indices.update(edge_to_tri_indices[cut_edge])

            triangle_indices.update(affected_tri_indices)

        # Step 2: Combine old and new vertices
        V_new = (
            np.vstack([V_orig, np.array(new_vertices)])
            if new_vertices
            else V_orig.copy()
        )
        labels_new = labels_orig + new_labels

        # Step 3: Subdivide triangles
        F_new = []
        for orig_index, tri in enumerate(F_orig):
            if orig_index not in triangle_indices:
                # Keep untouched triangle as-is
                face_index_mapping[orig_index] = [len(F_new)]
                F_new.append(tuple(tri))
                continue

            edge_to_new_vertex = {}
            for edge in triangle_edges(tri):
                norm_edge = normalize_edge(*edge)
                if norm_edge in edge_to_cutpoint_index:
                    edge_to_new_vertex[norm_edge] = edge_to_cutpoint_index[norm_edge]

            new_tris = split_triangle_topologically(tri, edge_to_new_vertex)
            new_face_indices = []
            for t in new_tris:
                new_index = len(F_new)
                new_face_indices.append(new_index)
                F_new.append(t)

            face_index_mapping[orig_index] = new_face_indices

        # Step 4: Canonicalize and validate
        F_new = self.canonicalize_faces(F_new)

        f_new_set = set(tuple(sorted(f)) for f in F_new)
        if len(f_new_set) != len(F_new):
            raise ValueError("Generated faces are not unique, there are duplicates.")

        return (
            PartitionableSpheroidTriangleMesh(V_new, np.array(F_new), labels_new),
            face_index_mapping,
        )

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import heapq
import math
from typing import Optional
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull, Delaunay, cKDTree


from py_3d_construct_lib.spherical_tools import (
    cartesian_to_spherical_jackson,
    coordinate_system_transform_to_matrix,
    spherical_to_cartesian_jackson,
    rotation_matrix_from_vectors,
    coordinate_system_transform,
)


def normalize_edge(a, b):
    return tuple(sorted((a, b)))


def triangle_edges(tri):
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


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


def tighten_vertex_path(vertex_path, edge_graph, segment_length, shorten_factor):
    """
    Given a path as a list of vertices, tighten it by replacing segments
    with shortest paths in the edge graph if they are significantly shorter.

    Raises:
        networkx.NetworkXNoPath if any tightening attempt fails to find a valid path.

    Returns:
        new_vertex_path: tightened list of vertices
    """
    path = vertex_path[:]
    i = 0

    while i < len(path) - segment_length:
        start = path[i]
        end = path[i + segment_length]

        shortest = nx.shortest_path(
            edge_graph, source=start, target=end, weight="weight"
        )

        original_segment = path[i : i + segment_length + 1]
        if (
            walk_length(shortest, edge_graph)
            < walk_length(original_segment, edge_graph) * shorten_factor
        ):
            path = path[:i] + shortest + path[i + segment_length + 1 :]
            # stay at same i to allow overlapping improvements
        else:
            i += 1

    assert is_valid_path(path, edge_graph), "Tightened path has broken edges"
    return path


@dataclass
class ConnectorHint:
    region_a: int
    region_b: int
    edge: tuple[int, int]  # vertex indices (v1, v2)

    edge_vector: np.ndarray  # unit vector from v1 to v2
    edge_centroid: np.ndarray  # midpoint of edge
    triangle_a_normal: np.ndarray  # normal vector of the triangle in region_a
    triangle_b_normal: np.ndarray  # normal vector of the triangle in region_b
    triangle_a: tuple[int, int, int]  # vertex indices
    triangle_b: tuple[int, int, int]  # vertex indices


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
            self.vertex_labels = np.array([i for i in range(len(self.vertices))])
        else:
            self.vertex_labels = vertex_labels
            assert len(self.vertex_labels) == len(
                self.vertices
            ), "Vertex labels must match the number of vertices"

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
    

    @staticmethod
    def create_shell_triangle_geometry(
        triangle_spherical_vertexes,
        sphere_center,
        shell_thickness,
        shrinkage=0.1,
        shrink_border=0,
    ):
        
        """
        Create a shell triangle geometry from the vertices of a triangle in spherical coordinates.
        This allows materializing a shell mesh, by creating solid triangle prisms from the spherical coordinates of the triangle vertexes.
        This is in spherical coordiates so that it is natural how to create the shell: The radius is offset by the shell_thickness, and the theta and phi are the same as the original triangle vertexes. 
        The resulting triangle prism for a whole mesh can then be fused to create a solid shell, which can for example be 3d-printed.
        """

        if len(triangle_spherical_vertexes) != 3:
            raise ValueError("triangle_spherical_vertexes must have 3 elements")

        for i in range(3):
            if len(triangle_spherical_vertexes[i]) != 3:
                raise ValueError(
                    "Each element of triangle_spherical_vertexes must have 3 elements (r, theta, phi)"
                )

        cartesian_vertexes = [
            spherical_to_cartesian_jackson(
                v, radius_offset=-shell_thickness, sphere_center=sphere_center
            )
            for v in triangle_spherical_vertexes
        ]
        outside_cartesian_vertexes = [
            spherical_to_cartesian_jackson(
                v, radius_offset=0, sphere_center=sphere_center
            )
            for v in triangle_spherical_vertexes
        ]

        # check if the vertexes are in the right order
        # if not, reverse the order

        if (
            np.cross(
                cartesian_vertexes[1] - cartesian_vertexes[0],
                cartesian_vertexes[2] - cartesian_vertexes[0],
            )[2]
            < 0
        ):
            cartesian_vertexes[1], cartesian_vertexes[2] = (
                cartesian_vertexes[2],
                cartesian_vertexes[1],
            )
            outside_cartesian_vertexes[1], outside_cartesian_vertexes[2] = (
                outside_cartesian_vertexes[2],
                outside_cartesian_vertexes[1],
            )

        centroid = np.sum(cartesian_vertexes, axis=0) / 6
        centroid += np.sum(outside_cartesian_vertexes, axis=0) / 6

        for i in range(3):
            cartesian_vertexes[i] = cartesian_vertexes[i] - shrinkage * (
                cartesian_vertexes[i] - centroid
            )
            outside_cartesian_vertexes[i] = outside_cartesian_vertexes[
                i
            ] - shrinkage * (outside_cartesian_vertexes[i] - centroid)

        # shrink with border
        if shrink_border > 0:
            cartesian_vertexes[0], cartesian_vertexes[1], cartesian_vertexes[2] = (
                shrink_triangle(
                    cartesian_vertexes[0],
                    cartesian_vertexes[1],
                    cartesian_vertexes[2],
                    border_width=shrink_border,
                )
            )
            (
                outside_cartesian_vertexes[0],
                outside_cartesian_vertexes[1],
                outside_cartesian_vertexes[2],
            ) = shrink_triangle(
                outside_cartesian_vertexes[0],
                outside_cartesian_vertexes[1],
                outside_cartesian_vertexes[2],
                border_width=shrink_border,
            )

        # Now use these six points to define a prism
        vertexes = {i: v for i, v in enumerate(cartesian_vertexes)}
        outside_vertexes = {i + 3: v for i, v in enumerate(outside_cartesian_vertexes)}
        cartesian_vertexes = {**vertexes, **outside_vertexes}
        maps = {
            "vertexes": cartesian_vertexes,
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
    def from_point_cloud(cls, point_cloud):

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

        return cls(vertices, triangles)

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


class MeshPartition:

    def __init__(self, mesh, face_to_region_map):
        self.mesh = mesh
        self.face_to_region_map = face_to_region_map
        self.boundary_edges = self._compute_boundary_edges()
        self.face_graph = self._build_face_graph()
        self.edge_graph = self._build_edge_graph()

    def _build_face_graph(self):
        G = nx.Graph()
        region_faces = self.face_to_region_map.keys()
        edge_to_faces = defaultdict(list)

        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for i in range(3):
                a, b = sorted((face[i], face[(i + 1) % 3]))
                edge_to_faces[(a, b)].append(f_idx)

        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                a, b = faces
                G.add_edge(a, b)

        return G

    def _build_edge_graph(self):
        G = nx.Graph()
        region_faces = self.face_to_region_map.keys()
        V = self.mesh.vertices

        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                dist = np.linalg.norm(V[a] - V[b])
                G.add_edge(a, b, weight=dist)

        return G

    def has_region_holes(self, region_id):
        region_faces = set(self.get_faces_of_region(region_id))
        edge_count = defaultdict(int)

        for f in region_faces:
            face = self.mesh.faces[f]
            for i in range(3):
                a, b = sorted((face[i], face[(i + 1) % 3]))
                edge_count[(a, b)] += 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        G = nx.Graph()
        for a, b in boundary_edges:
            G.add_edge(a, b)

        return nx.number_connected_components(G) > 1

    def construct_closed_path_from_vertices(
        self,
        region_face_indices: set[int],
        vertex_set: set[int],
    ) -> list[int]:
        """
        Given a set of vertex indices assumed to lie on a closed boundary,
        attempt to construct an ordered closed path using available edges
        in the region. If gaps exist, shortest paths are inserted.

        Returns:
            A list of vertex indices forming a closed path (first == last),
            covering all vertices in vertex_set and any minimal fillers.
        """
        from heapq import heappush, heappop

        V = self.mesh.vertices
        vertex_set = set(vertex_set)

        # Step 1: Build adjacency graph from region faces
        edge_graph = defaultdict(set)
        for face_idx in region_face_indices:
            face = self.mesh.faces[face_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                edge_graph[a].add(b)
                edge_graph[b].add(a)

        # Step 2: Identify subpaths in the input vertex set
        unused = set(vertex_set)
        subpaths = []

        while unused:
            start = unused.pop()
            path = [start]
            current = start
            while True:
                neighbors = edge_graph[current].intersection(unused)
                if not neighbors:
                    break
                next_v = neighbors.pop()
                path.append(next_v)
                unused.remove(next_v)
                current = next_v
            subpaths.append(path)

        # Step 3: Connect subpaths by shortest paths (greedy)
        def dijkstra_path(start, end):
            dist = {start: 0.0}
            prev = {}
            heap = [(0.0, start)]
            while heap:
                d, u = heappop(heap)
                if u == end:
                    break
                for v in edge_graph[u]:
                    nd = d + np.linalg.norm(V[u] - V[v])
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heappush(heap, (nd, v))
            # Reconstruct
            if end not in prev:
                raise ValueError(f"No path found from {start} to {end}")
            path = [end]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return list(reversed(path))

        while len(subpaths) > 1:
            best_pair = None
            best_cost = float("inf")
            best_connection = None
            for i in range(len(subpaths)):
                for j in range(i + 1, len(subpaths)):
                    a_tail, a_head = subpaths[i][0], subpaths[i][-1]
                    b_tail, b_head = subpaths[j][0], subpaths[j][-1]
                    # Try head to head, tail to tail, etc.
                    for u, udir in [(a_head, "f"), (a_tail, "r")]:
                        for v, vdir in [(b_head, "f"), (b_tail, "r")]:
                            try:
                                path = dijkstra_path(u, v)
                                cost = sum(
                                    np.linalg.norm(V[path[i + 1]] - V[path[i]])
                                    for i in range(len(path) - 1)
                                )
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pair = (i, j, udir, vdir)
                                    best_connection = path
                            except ValueError:
                                continue

            # Merge the best two
            i, j, udir, vdir = best_pair
            pi, pj = subpaths[i], subpaths[j]

            # Adjust directions
            if udir == "r":
                pi = pi[::-1]
            if vdir == "r":
                pj = pj[::-1]

            merged = pi + best_connection[1:] + pj
            new_subpaths = [
                subpaths[k] for k in range(len(subpaths)) if k not in [i, j]
            ]
            new_subpaths.append(merged)
            subpaths = new_subpaths

        final_path = subpaths[0]
        if final_path[0] != final_path[-1]:
            final_path.append(final_path[0])

        return final_path

    def is_region_contiguous(self, region_id):
        region_faces = self.get_faces_of_region(region_id)
        subgraph = self.face_graph.subgraph(region_faces)
        return nx.is_connected(subgraph)

    def _normalize_edge(self, a, b):
        return tuple(sorted((a, b)))

    def _compute_boundary_edges(self):
        edge_to_faces = defaultdict(set)
        for i, face in enumerate(self.mesh.faces):
            for j in range(3):
                edge = self._normalize_edge(face[j], face[(j + 1) % 3])
                edge_to_faces[edge].add(i)

        boundary_edges = defaultdict(set)

        for edge, adjacent_faces in edge_to_faces.items():
            if len(adjacent_faces) != 2:
                continue  # might be mesh border or degenerate

            face_list = list(adjacent_faces)
            region1 = self.face_to_region_map[face_list[0]]
            region2 = self.face_to_region_map[face_list[1]]

            if region1 != region2:
                key = frozenset([region1, region2])
                boundary_edges[key].add(edge)

        return boundary_edges

    def get_boundary_edges_of_region(self, region_id):
        edge_set = set()
        for key, edges in self.boundary_edges.items():
            if region_id in key:
                edge_set.update(edges)

        return edge_set

    def region_adjacency_graph(self):
        adj = defaultdict(set)
        for key in self.boundary_edges:
            a, b = tuple(key)
            adj[a].add(b)
            adj[b].add(a)
        return adj

    def get_faces_of_region(self, region):
        return [face for face, reg in self.face_to_region_map.items() if reg == region]

    def get_regions(self):
        return sorted(set(self.face_to_region_map.values()))

    def get_region_area(self, region):
        faces_of_region = self.get_faces_of_region(region)
        return sum(
            self.mesh.triangle_area(self.mesh.faces[face]) for face in faces_of_region
        )

    def get_submesh_maps(self, region_id):

        faces_of_region = self.get_faces_of_region(region_id)

        vertex_index_faces_of_region = [
            self.mesh.faces[face] for face in faces_of_region
        ]

        vertex_index_set = set()
        for face in vertex_index_faces_of_region:
            vertex_index_set.update(face)

        old_to_new_vertex_index_mapping = {}
        new_to_old_vertex_index_mapping = {}

        for new_vertex_index, old_vertex_index in enumerate(sorted(vertex_index_set)):
            old_to_new_vertex_index_mapping[old_vertex_index] = new_vertex_index
            new_to_old_vertex_index_mapping[new_vertex_index] = old_vertex_index

        new_faces = {}
        for new_face_index, face in enumerate(vertex_index_faces_of_region):
            new_faces[new_face_index] = tuple(
                old_to_new_vertex_index_mapping[vertex_index] for vertex_index in face
            )

        new_vertices = {}
        for (
            new_vertex_index,
            old_vertex_index,
        ) in new_to_old_vertex_index_mapping.items():
            new_vertices[new_vertex_index] = self.mesh.vertices[old_vertex_index]

        boundary_edges = self.get_boundary_edges_of_region(region_id)

        boundary_edges_new = {}
        for edge_number, edge in enumerate(boundary_edges):
            a, b = edge
            boundary_edges_new[edge_number] = (
                old_to_new_vertex_index_mapping[a],
                old_to_new_vertex_index_mapping[b],
            )

        maps = {
            "vertexes": new_vertices,
            "faces": new_faces,
            "boundary_edges": boundary_edges_new,
        }

        sorted_vertex_keys = list(sorted(maps["vertexes"].keys()))
        expected_vertex_keys = list(range(len(maps["vertexes"])))
        if not sorted_vertex_keys == expected_vertex_keys:
            raise ValueError(
                f"Vertex keys are not numbered correctly. Expected {expected_vertex_keys}, got {sorted_vertex_keys}"
            )

        return maps

    def split_region_by_cap(
        self,
        region_id: int,
        initial_seed_triangle_index: int,
        target_area_fraction: float,
        verbose: bool = False,
    ) -> "MeshPartition":
        """
        Splits a region by growing a spherical cap from a seed triangle centroid,
        and bisecting until the desired area fraction is reached.

        Parameters:
        - region_id: the region to split
        - initial_seed_triangle_index: triangle index (in global mesh) inside the region
        - target_area_fraction: area fraction to enclose in the cap (between 0 and 1)
        """
        if not (0.0 < target_area_fraction < 1.0):
            raise ValueError("target_area_fraction must be between 0 and 1.")

        mesh = self.mesh

        # Filter to relevant faces
        region_faces = [f for f, r in self.face_to_region_map.items() if r == region_id]
        region_set = set(region_faces)

        if initial_seed_triangle_index not in region_set:
            raise ValueError("Seed triangle must be part of the given region.")

        target_area = target_area_fraction * sum(
            mesh.triangle_area(mesh.faces[f]) for f in region_faces
        )

        seed_centroid = mesh.triangle_centroid(mesh.faces[initial_seed_triangle_index])
        R = rotation_matrix_from_vectors(seed_centroid, np.array([0, 0, 1]))

        rotated_coords = (R @ mesh.vertices.T).T
        rotated_r_theta_phi = np.array(
            [cartesian_to_spherical_jackson(v) for v in rotated_coords]
        )
        rotated_theta_phi = rotated_r_theta_phi[:, 1:3]

        # Bisection loop
        min_angle = 0
        max_angle = np.pi
        best_masked_faces = []

        while max_angle - min_angle > 1e-3:
            current_angle = (max_angle + min_angle) / 2

            # Vertex mask (by theta)
            vertex_mask = rotated_theta_phi[:, 0] <= current_angle

            # Find triangles where all 3 vertices are inside the cap
            masked_faces = []
            for f_idx in region_faces:
                tri = mesh.faces[f_idx]
                if all(vertex_mask[v] for v in tri):
                    masked_faces.append(f_idx)

            area = sum(mesh.triangle_area(mesh.faces[f]) for f in masked_faces)

            if verbose:
                print(
                    f"Cap angle: {current_angle:.4f}, area: {area:.4f}, target: {target_area:.4f}"
                )

            if area < target_area:
                min_angle = current_angle
            else:
                max_angle = current_angle
                best_masked_faces = masked_faces

        # Assign region ids
        new_region_id = max(self.get_regions()) + 1

        initial_faces_A = set(best_masked_faces)
        initial_faces_B = region_set - initial_faces_A

        print(f"Cap split with {len(initial_faces_A)} vs {len(initial_faces_B)} faces.")
        try:
            walk = self.extract_boundary_walk_between_face_sets(
                initial_faces_A, initial_faces_B
            )
            print(f"Extracted boundary walk: {walk}")
            tightened_A, tightened_B = self.tighten_boundary_walk(
                walk,
                segment_length=3,
                shorten_factor=0.99,
                allowed_faces=initial_faces_A | initial_faces_B,
            )
            print(f"Tightened A: {tightened_A}\nTightened B: {tightened_B}")
        except ValueError as e:
            if verbose:
                print(f"Skipping boundary tightening: {e}")
            tightened_A = initial_faces_A
            tightened_B = initial_faces_B

        final_face_to_region = dict(self.face_to_region_map)
        for f in tightened_A:
            final_face_to_region[f] = region_id
        for f in tightened_B:
            final_face_to_region[f] = new_region_id

        if verbose:
            a1 = sum(mesh.triangle_area(mesh.faces[f]) for f in tightened_A)
            a2 = sum(mesh.triangle_area(mesh.faces[f]) for f in tightened_B)
            print(
                f"Split region {region_id} → [{region_id}, {new_region_id}] (tightened) | "
                f"area A: {a1:.2f}, B: {a2:.2f}, target: {target_area:.2f}"
            )

        region_sizes = Counter(final_face_to_region.values())

        print(f"Final region sizes: {region_sizes}")

        return MeshPartition(mesh, final_face_to_region)

    def split_region_along_boundary_walk(
        self,
        region_id: int,
        target_area_fraction: float,
        seed_vertex: Optional[int] = None,
        verbose: bool = True,
    ) -> "MeshPartition":
        import heapq

        if not (0 < target_area_fraction < 1):
            raise ValueError("target_area_fraction must be between 0 and 1 (exclusive)")

        # Compute the absolute target area
        target_area = target_area_fraction * self.get_region_area(region_id)
        region_faces = set(self.get_faces_of_region(region_id))
        V = self.mesh.vertices

        # 1) Build edge→face map for this region
        edge_to_faces = defaultdict(list)
        for f in region_faces:
            verts = self.mesh.faces[f]
            for i in range(3):
                a, b = sorted((verts[i], verts[(i + 1) % 3]))
                edge_to_faces[(a, b)].append(f)

        # 2) Extract the closed boundary loop of vertices
        def extract_boundary_loop():
            boundary_edges = [e for e, fs in edge_to_faces.items() if len(fs) == 1]
            adj = defaultdict(list)
            for a, b in boundary_edges:
                adj[a].append(b)
                adj[b].append(a)

            start = boundary_edges[0][0]
            loop = [start]
            prev = None
            cur = start
            while True:
                nbrs = [v for v in adj[cur] if v != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if nxt == start:
                    loop.append(start)
                    break
                loop.append(nxt)
                prev, cur = cur, nxt
            return loop

        boundary_loop = extract_boundary_loop()
        # rotate so that boundary_loop[0] == seed_vertex
        if seed_vertex is None:
            seed_vertex = boundary_loop[0]
        i0 = boundary_loop.index(seed_vertex)
        boundary_loop = boundary_loop[i0:] + boundary_loop[1 : i0 + 1]  # keep closed

        # 3) Find opposite vertex: balances left/right boundary lengths
        def path_length(path):
            return sum(
                np.linalg.norm(V[path[i + 1]] - V[path[i]])
                for i in range(len(path) - 1)
            )

        def find_opposite(loop, seed):
            idx = loop.index(seed)
            best, bd = None, float("inf")
            n = len(loop) - 1  # last repeats seed
            for j in range(1, n):
                opp = loop[j]
                # two boundary branches
                right = loop[idx : j + 1] if j > idx else loop[idx:] + loop[: j + 1]
                left = loop[j : idx + 1] if j < idx else loop[j:] + loop[: idx + 1]
                diff = abs(path_length(left) - path_length(right))
                if diff < bd:
                    bd, best = diff, opp
            return best

        opposite = find_opposite(boundary_loop, seed_vertex)

        # 4) Precompute the two boundary branches
        idx_seed = boundary_loop.index(seed_vertex)
        idx_opp = boundary_loop.index(opposite)
        if idx_seed < idx_opp:
            branch_right = boundary_loop[idx_seed : idx_opp + 1]
            branch_left = boundary_loop[idx_opp:] + boundary_loop[: idx_seed + 1]
        else:
            branch_right = boundary_loop[idx_seed:] + boundary_loop[: idx_opp + 1]
            branch_left = boundary_loop[idx_opp : idx_seed + 1]

        # 5) Build interior graph for shortest path (only internal edges)
        internal_edges = [e for e, fs in edge_to_faces.items() if len(fs) == 2]
        graph = defaultdict(list)
        for a, b in internal_edges:
            d = np.linalg.norm(V[a] - V[b])
            graph[a].append((b, d))
            graph[b].append((a, d))

        def dijkstra(s, t):
            dist = {s: 0}
            prev = {}
            heap = [(0, s)]
            while heap:
                cd, u = heapq.heappop(heap)
                if u == t:
                    break
                if cd > dist[u]:
                    continue
                for v, w in graph[u]:
                    nd = cd + w
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            # reconstruct
            path = [t]
            while path[-1] != s:
                path.append(prev[path[-1]])
            return path[::-1]

        # Helper to walk a branch up to fraction t
        def walk_frac(path, t):
            tot = path_length(path)
            goal = t * tot
            acc = 0.0
            for i in range(len(path) - 1):
                step = np.linalg.norm(V[path[i + 1]] - V[path[i]])
                if acc + step >= goal:
                    return path[i + 1]
                acc += step
            return path[-1]

        new_id = max(self.get_regions()) + 1
        min_t, max_t = 0.0, 1.0
        best_assign = set()

        # 6) Bisection on t
        for _ in range(30):
            t = 0.5 * (min_t + max_t)
            lv = walk_frac(branch_left, t)
            rv = walk_frac(branch_right, t)

            # build the 3‐segment loop: seed→…→lv, lv→…→rv (interior), rv→…→seed
            iL = boundary_loop.index(lv)
            seg1 = boundary_loop[: iL + 1]  # seed→…→lv
            seg2 = dijkstra(lv, rv)  # lv→…→rv
            iR = boundary_loop.index(rv)
            seg3 = boundary_loop[iR:]  # rv→…→seed (last element is seed)

            # assemble cut‐loop edges
            def norm(e):
                return e if e[0] < e[1] else (e[1], e[0])

            cut = set()
            for seg in (seg1, seg2, seg3):
                for a, b in zip(seg, seg[1:]):
                    cut.add(norm((a, b)))

            # build triangle adjacency skipping cut edges
            tri_adj = defaultdict(list)
            for edge, fs in edge_to_faces.items():
                ne = norm(edge)
                if len(fs) == 2 and ne not in cut:
                    f1, f2 = fs
                    tri_adj[f1].append(f2)
                    tri_adj[f2].append(f1)

            # flood‐fill from a seed triangle
            def tri_for_vert(v):
                for f in region_faces:
                    if v in self.mesh.faces[f]:
                        return f
                raise RuntimeError

            start_tri = tri_for_vert(seed_vertex)
            visited = {start_tri}
            dq = deque([start_tri])
            while dq:
                u = dq.popleft()
                for w in tri_adj[u]:
                    if w not in visited:
                        visited.add(w)
                        dq.append(w)

            area = sum(self.mesh.triangle_area(self.mesh.faces[f]) for f in visited)
            if verbose:
                print(f" t={t:.4f}  areaA={area:.1f}  target={target_area:.1f}")

            if abs(area - target_area) < 1e-3 * target_area:
                best_assign = visited
                break
            if area < target_area:
                min_t = t
            else:
                max_t = t
            best_assign = visited

        # 7) Build new face→region map
        new_map = dict(self.face_to_region_map)
        for f in best_assign:
            new_map[f] = new_id

        if verbose:
            print(
                f" New region {new_id}: {len(best_assign)} faces vs {len(region_faces)-len(best_assign)}"
            )

        return MeshPartition(self.mesh, new_map)

    def is_path_well_formed(self, region_face_indices, path):
        if len(path) < 2:
            return False

        region_vertices = set()
        edge_set = set()
        for f_idx in region_face_indices:
            face = self.mesh.faces[f_idx]
            region_vertices.update(face)
            for i in range(3):
                a, b = sorted((face[i], face[(i + 1) % 3]))
                edge_set.add((a, b))

        for a, b in zip(path, path[1:]):
            if a not in region_vertices or b not in region_vertices:
                return False
            if (a, b) not in edge_set and (b, a) not in edge_set:
                return False

        return True

    def vertex_shortest_path(self, u, v):
        return nx.shortest_path(self.edge_graph, source=u, target=v, weight="weight")

    def region_view(self, region_id: int) -> "TransformedRegionView":
        """Return a view of the given region with identity transform."""
        return TransformedRegionView(self, region_id, transform=np.eye(4))

    def split_region_by_fibonacci_plane(
        self,
        region_id: int,
        target_area_fraction: float,
        samples: int = 300,
        verbose: bool = False,
    ) -> "MeshPartition":
        """
        Split a region by rotating it via Fibonacci sphere directions and slicing along the x=0 plane.
        Chooses the orientation that results in the closest match to the target area fraction.
        """

        if not (0.0 < target_area_fraction < 1.0):
            raise ValueError("target_area_fraction must be between 0 and 1.")

        view = self.region_view(region_id)
        directions = fibonacci_sphere(samples)

        best_diff = float("inf")
        best_split = None

        for d in directions:
            R = rotation_matrix_from_vectors(d, np.array([1.0, 0.0, 0.0]))  # x-axis
            A = np.eye(4)
            A[:3, :3] = R
            rotated_view = view.apply_transform(A)

            V, F, _ = rotated_view.get_transformed_vertices_faces_boundary_edges()

            area_total = 0.0
            area_A = 0.0
            faces_A = []
            faces_B = []

            for i, face in enumerate(F):
                verts = V[face]
                A_face = 0.5 * np.linalg.norm(
                    np.cross(verts[1] - verts[0], verts[2] - verts[0])
                )
                area_total += A_face
                if all(v[0] > 0 for v in verts):
                    faces_A.append(i)
                    area_A += A_face
                else:
                    faces_B.append(i)

            area_fraction = area_A / area_total if area_total > 0 else 0
            diff = abs(area_fraction - target_area_fraction)

            if diff < best_diff:
                best_diff = diff
                best_split = (faces_A, faces_B)

        # --- Build new face_to_region_map ---
        region_faces = self.get_faces_of_region(region_id)
        region_to_global_face = {i: f for i, f in enumerate(region_faces)}

        new_region_id = max(self.get_regions()) + 1
        new_face_to_region = dict(self.face_to_region_map)

        for i in best_split[0]:
            new_face_to_region[region_to_global_face[i]] = (
                region_id  # stay in original region
            )
        for i in best_split[1]:
            new_face_to_region[region_to_global_face[i]] = (
                new_region_id  # move to new region
            )

        if verbose:
            a1 = sum(
                self.mesh.triangle_area(self.mesh.faces[region_to_global_face[i]])
                for i in best_split[0]
            )
            a2 = sum(
                self.mesh.triangle_area(self.mesh.faces[region_to_global_face[i]])
                for i in best_split[1]
            )
            print(
                f"Split region {region_id} → [{region_id}, {new_region_id}] | "
                f"area A: {a1:.2f}, B: {a2:.2f}, diff: {best_diff:.4f}"
            )

        region_sizes = Counter(new_face_to_region.values())

        print(f"Final region sizes: {region_sizes}")

        return MeshPartition(self.mesh, new_face_to_region)

    def extract_boundary_walk_between_face_sets(self, faces_a, faces_b):
        """
        Given two disjoint sets of face indices, extract the boundary walk separating them.

        Returns:
            walk: list of ordered edges [(v0, v1), (v1, v2), ...]
        """
        mesh = self.mesh
        # Collect which face owns which edge
        edge_to_faces = defaultdict(list)

        for f_idx in faces_a:
            face = mesh.faces[f_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_to_faces[edge].append(("A", f_idx))

        for f_idx in faces_b:
            face = mesh.faces[f_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_to_faces[edge].append(("B", f_idx))


        # Boundary edges: shared exactly between one face in A and one in B
        boundary_edges = []
        for edge, owners in edge_to_faces.items():
            if len(owners) == 2 and {owners[0][0], owners[1][0]} == {"A", "B"}:
                boundary_edges.append(edge)

        print(f"Boundary edges: {boundary_edges}")

        # Build adjacency graph of boundary edges
        vertex_adj = defaultdict(list)
        for a, b in boundary_edges:
            vertex_adj[a].append(b)
            vertex_adj[b].append(a)

        # Find endpoints (degree 1 vertices)
        endpoints = [v for v, neighbors in vertex_adj.items() if len(neighbors) == 1]
        if len(endpoints) == 0:
            print(f"Boundary is closed loop")
            start = next(iter(vertex_adj))  # just pick any vertex
        elif len(endpoints) > 2:
            raise ValueError(
                "Boundary walk has multiple disconnected components or branches."
            )
        else:
            start = endpoints[0]


        # Traverse the walk

        walk = []
        visited_edges = set()
        current = start

        while True:
            neighbors = [
                n
                for n in vertex_adj[current]
                if (min(current, n), max(current, n)) not in visited_edges
            ]
            if not neighbors:
                break
            next_v = neighbors[0]
            edge = (min(current, next_v), max(current, next_v))
            walk.append((current, next_v))
            visited_edges.add(edge)
            current = next_v

        return walk

    def tighten_boundary_walk(
        self, walk, allowed_faces, segment_length, shorten_factor
    ):
        """
        Tightens a walk by replacing segments with shortest paths in the edge graph,
        represented as an ordered list of vertices. Then flood-fills the triangle mesh
        into two disjoint regions using the walk as a separator.

        Returns:
            (faces_a, faces_b): sets of triangle indices assigned to each side of the walk.
        """
        mesh = self.mesh

        # --- Build edge graph from allowed faces ---
        edge_graph = nx.Graph()
        for f_idx in allowed_faces:
            tri = mesh.faces[f_idx]
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                dist = np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])
                edge_graph.add_edge(a, b, weight=dist)

        # --- Convert original walk to ordered vertex list ---
        vertex_walk = [walk[0][0], walk[0][1]]
        for edge in walk[1:]:
            if edge[0] == vertex_walk[-1]:
                vertex_walk.append(edge[1])
            elif edge[1] == vertex_walk[-1]:
                vertex_walk.append(edge[0])
            else:
                raise ValueError(f"Non-contiguous edge in original walk: {edge}")

        def is_valid_vertex_walk(vertex_list):
            return all(
                edge_graph.has_edge(vertex_list[i], vertex_list[i + 1])
                for i in range(len(vertex_list) - 1)
            )

        def walk_length(vertex_list):
            return sum(
                edge_graph[vertex_list[i]][vertex_list[i + 1]]["weight"]
                for i in range(len(vertex_list) - 1)
            )

        original_walk_length = walk_length(vertex_walk)
        print(f"Original vertex walk: {vertex_walk}, length: {original_walk_length}")
        
        # --- Shorten walk by replacing segments with shortest paths ---
        i = 0
        while i < len(vertex_walk) - segment_length - 1:
            start = vertex_walk[i]
            end = vertex_walk[i + segment_length]
            original_segment = vertex_walk[i : i + segment_length + 1]

            path = nx.shortest_path(
                edge_graph, source=start, target=end, weight="weight"
            )
            if walk_length(path) < walk_length(original_segment) * shorten_factor:
                vertex_walk = (
                    vertex_walk[:i] + path + vertex_walk[i + segment_length + 1 :]
                )
                i += len(path) - 1
            else:
                i += 1

            assert is_valid_vertex_walk(
                vertex_walk
            ), "Tightened vertex walk is not edge-connected."

        print(f"Vertex walk after: {vertex_walk}")
        # --- Extract edge set for forbidden boundary ---
        forbidden_edges = {
            tuple(sorted((vertex_walk[i], vertex_walk[i + 1])))
            for i in range(len(vertex_walk) - 1)
        }

        # --- Build face adjacency map ---
        face_edges = {}
        face_adjacency = defaultdict(set)
        for f_idx in allowed_faces:
            tri = mesh.faces[f_idx]
            edges = {tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3)}
            face_edges[f_idx] = edges

        allowed_list = list(allowed_faces)
        for i, f1 in enumerate(allowed_list):
            for j in range(i + 1, len(allowed_list)):
                f2 = allowed_list[j]
                if face_edges[f1] & face_edges[f2]:
                    face_adjacency[f1].add(f2)
                    face_adjacency[f2].add(f1)

        # --- Find seed triangles adjacent to the first walk edge ---
        first_edge = tuple(sorted((vertex_walk[0], vertex_walk[1])))
        edge_to_faces = defaultdict(set)
        for f_idx in allowed_faces:
            for e in face_edges[f_idx]:
                edge_to_faces[e].add(f_idx)

        seeds = list(edge_to_faces[first_edge])
        if len(seeds) != 2:
            raise ValueError("Cannot find two seed triangles for the first edge.")
        seed_a, seed_b = seeds

        # --- Flood fill using triangle adjacency, stopping at forbidden edges ---
        visited = set()

        def flood_fill(seed, forbidden_faces):
            region = set()
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                if current in region or current in visited:
                    continue
                visited.add(current)
                region.add(current)
                for neighbor in face_adjacency[current]:
                    shared = face_edges[current] & face_edges[neighbor]
                    if any(e in forbidden_edges for e in shared):
                        continue
                    if neighbor in forbidden_faces:
                        raise ValueError(
                            f"Flood fill leaked into forbidden face {neighbor}"
                        )

                    queue.append(neighbor)
            return region

        region_a = flood_fill(seed_a, forbidden_faces={seed_b})
        region_b = flood_fill(seed_b, forbidden_faces={seed_a})

        if not region_a or not region_b:
            raise ValueError("Flood fill failed: one region is empty.")

        assert region_a.isdisjoint(region_b), "Regions are not disjoint."
        assert (
            region_a | region_b == allowed_faces
        ), "Flood fill did not cover all allowed faces."

        print(f"Tightened walk length: {walk_length(vertex_walk)}, original: {original_walk_length}")
        print(f"Tightened vertex walk: {vertex_walk}")
        print(f"Region A: {len(region_a)} faces, Region B: {len(region_b)} faces")

        # Use lexicographic heuristic for naming A vs B
        if min(region_a) < min(region_b):
            return region_a, region_b
        else:
            return region_b, region_a

    # def tighten_boundary_walk(
    #     self, walk, allowed_faces, segment_length=5, shorten_factor=0.9
    # ):
    #     """
    #     Tightens a walk by replacing segments with shortest paths in the edge graph.

    #     Returns:
    #         (faces_a, faces_b): sets of triangle indices assigned to each side of the walk.
    #         Classification is done via triangle adjacency flood-fill seeded on both sides.
    #     """
    #     mesh = self.mesh

    #     # --- Step 1: Build edge graph from allowed_faces ---
    #     edge_graph = nx.Graph()
    #     for f_idx in allowed_faces:
    #         face = mesh.faces[f_idx]
    #         for i in range(3):
    #             a, b = face[i], face[(i + 1) % 3]
    #             dist = np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])
    #             edge_graph.add_edge(a, b, weight=dist)

    #     def walk_length(edges):
    #         return sum(edge_graph[u][v]["weight"] for u, v in edges)

    #     # --- Step 2: Walk tightening (preserving direction) ---
    #     def append_chain(walk_accum, chain):
    #         last = walk_accum[-1][1]
    #         for u, v in chain:
    #             if u == last:
    #                 walk_accum.append((u, v))
    #             elif v == last:
    #                 walk_accum.append((v, u))
    #             else:
    #                 raise ValueError(f"Cannot connect edge {(u, v)} to walk ending at {last}")

    #     new_walk = [walk[0]]  # start with the first edge in directed form
    #     i = 1
    #     n = len(walk)
    #     while i < n:
    #         rem = n - i
    #         if rem >= segment_length:
    #             segment = walk[i:i + segment_length]
    #             start = new_walk[-1][1]
    #             end = segment[-1][1]

    #             try:
    #                 path = nx.shortest_path(edge_graph, source=start, target=end, weight="weight")
    #             except nx.NetworkXNoPath:
    #                 print(f"No path found between {start} and {end}, falling back to original edge.")
    #                 append_chain(new_walk, [walk[i]])
    #                 i += 1
    #                 continue

    #             if path[0] != new_walk[-1][1]:
    #                 path = path[::-1]  # fix direction

    #             tightened = [(path[j], path[j + 1]) for j in range(len(path) - 1)]

    #             if walk_length(tightened) < walk_length(segment) * shorten_factor:
    #                 append_chain(new_walk, tightened)
    #                 i += segment_length
    #             else:
    #                 append_chain(new_walk, [walk[i]])
    #                 i += 1
    #         else:
    #             append_chain(new_walk, walk[i:])
    #             break

    #     print(f"Tightened walk length: {walk_length(new_walk)}")
    #     print(f"Original walk: {walk}\nTightened walk: {new_walk}")

    #     # --- Step 3: Triangle adjacency map (for flood fill) ---
    #     face_adjacency = defaultdict(set)
    #     face_edges = {}

    #     for f_idx in allowed_faces:
    #         tri = mesh.faces[f_idx]
    #         edges = {tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3)}
    #         face_edges[f_idx] = edges

    #     allowed_face_list = list(allowed_faces)
    #     for i, f_idx in enumerate(allowed_face_list):
    #         for j in range(i + 1, len(allowed_face_list)):
    #             g_idx = allowed_face_list[j]
    #             if face_edges[f_idx] & face_edges[g_idx]:
    #                 face_adjacency[f_idx].add(g_idx)
    #                 face_adjacency[g_idx].add(f_idx)

    #     def is_connected_walk(walk):
    #         """
    #         Verifies that the walk is a single connected edge chain (open or closed),
    #         with no branches, islands, or multiple disjoint chains.
    #         """
    #         vertex_degree = defaultdict(int)
    #         for a, b in walk:
    #             vertex_degree[a] += 1
    #             vertex_degree[b] += 1

    #         degrees = list(vertex_degree.values())
    #         deg1 = degrees.count(1)
    #         deg2 = degrees.count(2)
    #         deg_more = [d for d in degrees if d > 2]

    #         # A valid walk must have only degree 2 vertices, except optionally two endpoints (degree 1)
    #         if deg_more:
    #             return False
    #         return (deg1 == 2 or deg1 == 0) and (deg1 + deg2 == len(degrees))

    #     assert is_connected_walk(new_walk), "Tightened walk must be a connected edge chain."

    #     # --- Step 4: Find the two triangles adjacent to the first walk edge ---
    #     first_edge = tuple(sorted(new_walk[0]))
    #     edge_to_faces = defaultdict(set)
    #     for f_idx in allowed_faces:
    #         tri = mesh.faces[f_idx]
    #         for i in range(3):
    #             edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
    #             edge_to_faces[edge].add(f_idx)

    #     # --- Step 5: Use first walk edge to get adjacent triangles ---
    #     first_edge = tuple(sorted(new_walk[0]))
    #     adjacent_faces = list(edge_to_faces[first_edge])

    #     if len(adjacent_faces) != 2:
    #         raise ValueError(f"Edge {first_edge} is not shared by exactly two triangles.")

    #     seed_alpha, seed_beta = adjacent_faces

    #     # --- Step 5: Flood fill from both sides, not crossing walk edges ---
    #     forbidden_edges = {tuple(sorted(e)) for e in new_walk}
    #     visited_global = set()

    #     def flood_fill(seed, forbidden_edges, visited_global, allowed_faces, forbidden_faces):
    #         visited = set()
    #         queue = deque([seed])
    #         while queue:
    #             current = queue.popleft()
    #             if current in visited or current in visited_global:
    #                 continue
    #             if current not in allowed_faces:
    #                 continue

    #             visited.add(current)
    #             visited_global.add(current)

    #             # --- Correctly indent this loop ---
    #             for neighbor in face_adjacency[current]:
    #                 if neighbor not in allowed_faces:
    #                     continue

    #                 # only block if *any* of the shared edges is a forbidden boundary
    #                 shared_edges = face_edges[current] & face_edges[neighbor]
    #                 if any(e in forbidden_edges for e in shared_edges):
    #                     continue

    #                 if neighbor in forbidden_faces:
    #                     raise ValueError(
    #                         f"Flood fill encountered a forbidden face: {neighbor}"
    #                     )
    #                 queue.append(neighbor)

    #         return visited
    #     region_alpha = flood_fill(seed_alpha, forbidden_edges, visited_global, allowed_faces, forbidden_faces={seed_beta})
    #     region_beta = flood_fill(seed_beta, forbidden_edges, visited_global, allowed_faces, forbidden_faces={seed_alpha})
    #     print(f"Region Alpha: {len(region_alpha)} faces, Region Beta: {len(region_beta)} faces")

    #     if not region_alpha or not region_beta:
    #         raise ValueError("One side of the boundary walk could not be flood-filled. "
    #                          "Check for isolated seeds or fully enclosed regions.")

    #     assert region_alpha.isdisjoint(region_beta), "Regions A and B must be disjoint."

    #     assert region_alpha | region_beta == allowed_faces,  "Flood fill must cover all allowed faces."

    #     # --- Step 6: Use overlap heuristic to decide who is A or B ---
    #     # Default to whichever has smaller face index if no better info
    #     if min(region_alpha) < min(region_beta):
    #         faces_a, faces_b = region_alpha, region_beta
    #     else:
    #         faces_a, faces_b = region_beta, region_alpha

    #     assert faces_a.isdisjoint(faces_b), "Regions A and B must be disjoint."
    #     print(f"Faces A: {len(faces_a)}, Faces B: {len(faces_b)}")

    #     return faces_a, faces_b

    def split_region_by_polar_oriented_plane(
        self,
        region_id: int,
        target_area_fraction: float,
        phi: float = 0.0,
        steps: int = 50,
        verbose: bool = False,
    ) -> "MeshPartition":
        """
        Rotate region to align its mean direction with Z+, apply additional Z-rotation by `phi`,
        and then split it by sweeping an x-cut plane until `target_area_fraction` is reached.

        Parameters:
        - region_id: ID of region to split
        - target_area_fraction: desired area ratio (between 0 and 1)
        - phi: extra rotation angle (in radians) around the Z-axis
        - steps: number of candidate x-cuts to try
        """
        if not (0.0 < target_area_fraction < 1.0):
            raise ValueError("target_area_fraction must be between 0 and 1.")

        view = self.region_view(region_id)

        # Step 1: Compute average direction of region (mean of vertex positions, normalized)
        V, F, _ = view.get_transformed_vertices_faces_boundary_edges()
        mean_vec = V.mean(axis=0)
        mean_vec /= np.linalg.norm(mean_vec)

        # Step 2: Rotate mean_vec to point "up" (to Z+)
        R_align = rotation_matrix_from_vectors(mean_vec, np.array([0, 0, 1]))
        A_align = np.eye(4)
        A_align[:3, :3] = R_align

        # Step 3: Rotate around Z-axis by phi
        c, s = np.cos(phi), np.sin(phi)
        R_phi = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )
        A_phi = np.eye(4)
        A_phi[:3, :3] = R_phi

        # Combined transform
        A = A_phi @ A_align
        rotated_view = view.apply_transform(A)

        V_rot, F_rot, _ = rotated_view.get_transformed_vertices_faces_boundary_edges()
        region_faces = self.get_faces_of_region(region_id)
        region_to_global_face = {i: f for i, f in enumerate(region_faces)}

        min_x = np.min(V_rot[:, 0])
        max_x = np.max(V_rot[:, 0])

        best_diff = float("inf")
        best_split = None

        low = min_x
        high = max_x
        best_diff = float("inf")
        best_split = None
        epsilon = 1e-6  # convergence threshold

        while high - low > epsilon:
            x_cut = 0.5 * (low + high)

            faces_A, faces_B = [], []
            area_A, area_total = 0.0, 0.0

            for j, face in enumerate(F_rot):
                verts = V_rot[face]
                area = 0.5 * np.linalg.norm(
                    np.cross(verts[1] - verts[0], verts[2] - verts[0])
                )
                area_total += area

                triangle_centroid = np.mean(verts, axis=0)
                if triangle_centroid[0] > x_cut:
                    faces_A.append(j)
                    area_A += area
                else:
                    faces_B.append(j)

            area_fraction = area_A / area_total if area_total > 0 else 0
            diff = abs(area_fraction - target_area_fraction)

            if diff < best_diff:
                best_diff = diff
                best_split = (faces_A, faces_B)

            # Update interval
            if area_fraction < target_area_fraction:
                high = x_cut
            else:
                low = x_cut
        # Build new region mapping
        new_region_id = max(self.get_regions()) + 1

        # Reconstruct initial face sets
        initial_faces_A = {region_to_global_face[i] for i in best_split[0]}
        initial_faces_B = {region_to_global_face[i] for i in best_split[1]}

        print(f"Best split at x={0.5 * (low + high):.4f} with diff={best_diff:.4f}")
        print(f"Faces A: {initial_faces_A},\nFaces B: {initial_faces_B}")

        # Step 4: Tighten the boundary between them
        try:
            walk = self.extract_boundary_walk_between_face_sets(
                initial_faces_A, initial_faces_B
            )
            print(f"Extracted boundary walk: {walk}")
            tightened_A, tightened_B = self.tighten_boundary_walk(
                walk, initial_faces_A | initial_faces_B,segment_length=3, shorten_factor=0.98
            )
            print(f"Tightened A: {tightened_A},\nTightened B: {tightened_B}")
        except ValueError as e:
            if verbose:
                print(f"Skipping boundary tightening: {e}")

            tightened_A = initial_faces_A
            tightened_B = initial_faces_B

        # Step 5: Construct final face_to_region_map
        final_face_to_region = dict(self.face_to_region_map)
        for f in tightened_A:
            final_face_to_region[f] = region_id
        for f in tightened_B:
            final_face_to_region[f] = new_region_id

        if verbose:
            a1 = sum(self.mesh.triangle_area(self.mesh.faces[f]) for f in tightened_A)
            a2 = sum(self.mesh.triangle_area(self.mesh.faces[f]) for f in tightened_B)
            print(
                f"Split region {region_id} → [{region_id}, {new_region_id}] (tightened) | "
                f"area A: {a1:.2f}, B: {a2:.2f}, diff: {best_diff:.4f}"
            )

        region_sizes = Counter(final_face_to_region.values())

        print(f"Final region sizes: {region_sizes}")

        return MeshPartition(self.mesh, final_face_to_region)


class TransformedRegionView:
    def __init__(
        self,
        partition: MeshPartition,
        region_id: int,
        transform: Optional[np.ndarray] = None,
    ):
        self.partition = partition
        self.region_id = region_id
        self.transform = transform if transform is not None else np.eye(4)

    def apply_transform(self, mat4x4: np.ndarray):
        """Returns a new view with the composed transformation applied."""
        new_transform = mat4x4 @ self.transform
        return TransformedRegionView(self.partition, self.region_id, new_transform)

    def rotated(
        self, angle: float, axis: np.ndarray = None, center: np.ndarray = None
    ) -> "TransformedRegionView":
        """
        Return a new TransformedRegionView rotated around a given axis and center point.

        Parameters:
        -----------
        angle : float
            Rotation angle in radians.
        axis : np.ndarray
            3-element vector defining the rotation axis.
        center : np.ndarray
            3-element point around which the rotation is applied.
        """
        if axis is None:
            axis = np.array([0, 0, 1])
        if center is None:
            center = np.array([0, 0, 0])
        if isinstance(axis, list) or isinstance(axis, tuple):
            axis = np.array(axis)
        if isinstance(center, list) or isinstance(center, tuple):
            center = np.array(center)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c

        # Rotation matrix using Rodrigues' formula
        R = np.array(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ]
        )

        # Compose affine 4x4 rotation matrix around `center`
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = center - R @ center

        return self.apply_transform(A)

    def translated(self, x, y=None, z=None) -> "TransformedRegionView":
        """
        Return a new TransformedRegionView translated by (x, y, z).

        Parameters:
        -----------
        x : float or array-like
            X component of translation or a 3-element vector.
        y : float, optional
            Y component of translation.
        z : float, optional
            Z component of translation.
        """
        if isinstance(x, (list, tuple, np.ndarray)) and y is None and z is None:
            vec = np.array(x, dtype=float)
        else:
            vec = np.array([x, y, z], dtype=float)

        T = np.eye(4)
        T[:3, 3] = vec
        return self.apply_transform(T)

    def get_transformed_vertices_faces_boundary_edges(self):
        """Return transformed vertex array and face index list."""
        maps = self.partition.get_submesh_maps(self.region_id)

        V = np.array([maps["vertexes"][i] for i in sorted(maps["vertexes"])])
        F = np.array([maps["faces"][i] for i in sorted(maps["faces"])])
        E = np.array(
            [maps["boundary_edges"][i] for i in sorted(maps["boundary_edges"])]
        )

        vertex_indices_in_edges = set()
        for a, b in E:
            vertex_indices_in_edges.add(a)
            vertex_indices_in_edges.add(b)

        assert all(
            j < len(V) for j in vertex_indices_in_edges
        ), "Vertex indices in edges are out of bounds"

        # Apply affine transformation to homogeneous coords
        V_homo = np.concatenate([V, np.ones((len(V), 1))], axis=1)
        V_transformed = (self.transform @ V_homo.T).T[:, :3]

        return V_transformed, F, E

    def get_transformed_materialized_shell_maps(self, **kwargs):
        """
        Return a dict of shell maps (face_id -> vertex/face map),
        where all vertex coordinates are transformed by the current affine matrix.
        """
        maps = self.partition.mesh.calculate_materialized_shell_maps(**kwargs)
        region_faces = self.partition.get_faces_of_region(self.region_id)

        result = {}
        for face_id in region_faces:
            face_map = maps[face_id]
            V = face_map["vertexes"]
            V_arr = np.array([V[k] for k in sorted(V)])
            V_homo = np.concatenate([V_arr, np.ones((len(V_arr), 1))], axis=1)
            V_transformed = (self.transform @ V_homo.T).T[:, :3]
            transformed_vertexes = {k: v for k, v in zip(sorted(V), V_transformed)}
            result[face_id] = {
                "vertexes": transformed_vertexes,
                "faces": face_map["faces"],
            }

        return result

    def lay_flat(self, definition_of_low: float = 1) -> "TransformedRegionView":
        """
        Return a new TransformedRegionView where the region is rotated and translated so that
        one of the low triangles lies flat on the XY plane.
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()

        z_min = np.min(V[:, 2])
        z_max = np.max(V[:, 2])
        low_thresh = z_min + definition_of_low * (z_max - z_min)

        low_faces = [f for f in F if any(V[i][2] <= low_thresh for i in f)]
        if not low_faces:
            raise ValueError("No low faces found.")
        else:
            print(f"Low faces found: {low_faces}")

        def normal(face):
            a, b, c = [V[i] for i in face]
            n = np.cross(b - a, c - a)
            return n / np.linalg.norm(n)

        good_faces = []
        for face in low_faces:

            a, b, c = [V[i] for i in face]

            print(f"Low face: {face}, vertices: {a}, {b}, {c}")
            # 2) Compute centroid pivot
            centroid = (a + b + c) / 3

            # 3) Build rotation R that carries face_normal → [0,0,1]
            fn = -normal(face)
            target = np.array([0.0, 0.0, 1.0])
            R3 = rotation_matrix_from_vectors(fn, target)  # your existing utility

            # 4) Assemble the full affine A = T_z * T( +centroid ) * R * T( -centroid )
            # 4×4 identity:
            A = np.eye(4)

            # T1 = translate(-centroid)
            T1 = np.eye(4)
            T1[:3, 3] = -centroid

            # R4 = rotation about origin
            R4 = np.eye(4)
            R4[:3, :3] = R3

            # T2 = translate(+centroid)
            T2 = np.eye(4)
            T2[:3, 3] = centroid

            # Combine: first T1, then R4, then T2
            M = T2 @ R4 @ T1

            pts_face_m = [(M @ np.hstack([v, 1]).T)[:3] for v in (a, b, c)]
            z_face = (
                sum(p[2] for p in pts_face_m) / 3.0
            )  # they should all be equal up to FP noise

            # Build T3 to drop *that* face to Z=0
            T3 = np.eye(4)
            T3[2, 3] = -z_face

            # Final composite
            A = T3 @ M

            to_flatten = [a, b, c]
            to_flatten_transformed = [(A @ np.hstack([v, 1]).T) for v in to_flatten]

            for v in to_flatten_transformed:
                if not np.isclose(v[2], 0, atol=1e-5):
                    print(f"WARNING: vertex not flat: {v}")

            # check if at least one triangle lies flat on the floor
            new_view = self.apply_transform(A)

            V_flat, F_flat, _ = new_view.get_transformed_vertices_faces_boundary_edges()

            # # no vertex should be below the floor
            if not np.any(V_flat[:, 2] < -1e-5):
                print(
                    f"Found  good face: {face}, vertices: {V_flat[F_flat[face]]}, all other vertices are above the floor"
                )
                good_faces.append({"face": face, "transform": A})

        if not good_faces:
            print(f"******** NO GOOD FACES FOUND ********")
            raise ValueError("No good faces found.")

        A = good_faces[0]["transform"]

        new_view = self.apply_transform(A)

        V_flat, F_flat, _ = new_view.get_transformed_vertices_faces_boundary_edges()

        found_face_number = None
        for face_number, faces in enumerate(F_flat):
            a, b, c = [V_flat[i] for i in faces]
            if np.isclose(a[2], 0) and np.isclose(b[2], 0) and np.isclose(c[2], 0):
                found_face_number = face_number
                break

        if found_face_number is None:
            raise ValueError("No flat face found after transformation.")

        else:
            print(
                f"Flat face found: {found_face_number}, vertex_indixes: {F_flat[found_face_number]} vertices: {V_flat[F_flat[found_face_number]]}, selected face_number: {face}"
            )

        return new_view

    def check_printability(self, overhang_threshold_deg: float = 45.0):
        """
        Check 3D printability of the region: identify triangles with too-steep overhangs.

        Parameters:
        -----------
        overhang_threshold_deg : float
            Maximum allowed angle (in degrees) between the triangle normal and the Z-axis.
            Triangles with greater angles are considered non-printable.

        Returns:
        --------
        dict with:
            - total_area: float
            - printable_area: float
            - unprintable_area: float
            - bad_faces: list of (face_index, angle_in_degrees)
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()
        z_axis = np.array([0, 0, 1])
        threshold_rad = np.radians(overhang_threshold_deg)

        def triangle_area(a, b, c):
            return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

        total_area = 0.0
        unprintable_area = 0.0
        bad_faces = []

        for idx, face in enumerate(F):
            a, b, c = V[face[0]], V[face[1]], V[face[2]]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-8:
                continue  # skip degenerate
            n /= np.linalg.norm(n)
            angle = np.arccos(np.clip(np.dot(n, z_axis), -1.0, 1.0))
            area = triangle_area(a, b, c)
            total_area += area

            if angle > threshold_rad:
                unprintable_area += area
                bad_faces.append((idx, np.degrees(angle)))

        return {
            "total_area": total_area,
            "printable_area": total_area - unprintable_area,
            "unprintable_area": unprintable_area,
            "bad_faces": bad_faces,
            "bad_fraction": unprintable_area / total_area if total_area > 0 else 0.0,
        }

    def find_overhanging_boundary_edges(
        self,
        angle_threshold_deg=45,
        vertical_edge_tolerance_deg=10,
        triangle_downward_threshold_deg=45,
    ) -> list[tuple[int, int]]:
        """
        Find boundary edges that need support due to overhang.

        An edge is overhanging if its triangle's "inward" direction (orthogonal to the edge and triangle normal),
        properly disambiguated to point inward, points upward more than `angle_threshold_deg` from horizontal.

        Parameters:
        -----------
        angle_threshold_deg : float
            Maximum allowable angle from horizontal. Edges exceeding this upward are marked.
        vertical_edge_tolerance_deg : float
            Edges more vertical than this angle are skipped entirely.
        """
        angle_threshold_sin = -np.sin(np.radians(triangle_downward_threshold_deg))
        vertical_edge_cos = np.cos(np.radians(vertical_edge_tolerance_deg))

        V_trans, region_faces, boundary_edges = (
            self.get_transformed_vertices_faces_boundary_edges()
        )
        V = V_trans
        result = []

        print(
            f"Found {len(boundary_edges)} boundary edges and {len(region_faces)} faces in region {self.region_id}"
        )

        for a, b in boundary_edges:
            # Find triangle in region that contains this edge
            tri = next((face for face in region_faces if a in face and b in face), None)
            if tri is None:
                raise ValueError(f"Edge {a}-{b} not found in region faces")

            c = next(i for i in tri if i not in (a, b))
            va, vb, vc = V[a], V[b], V[c]

            edge_vec = vb - va
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-8:
                continue  # degenerate edge
            edge_dir = edge_vec / edge_len

            # Skip nearly vertical edges — they print fine
            if abs(edge_dir[2]) > vertical_edge_cos:
                print(
                    f"Edge {a}-{b} is nearly vertical (z = {edge_dir[2]:.3f}), skipping."
                )
                continue

            triangle_normal = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
            triangle_normal /= np.linalg.norm(triangle_normal)

            # Check if triangle is downward-facing
            downward_problematic = False
            if triangle_normal[2] < angle_threshold_sin:
                print(f"Triangle {tri} is downward-facing")
                downward_problematic = True

            inward = np.cross(triangle_normal, edge_dir)
            inward /= np.linalg.norm(inward)

            edge_center = (va + vb) / 2
            tri_centroid = (va + vb + vc) / 3

            to_centroid = tri_centroid - edge_center
            to_centroid /= np.linalg.norm(to_centroid)

            # Flip inward if it points outward
            if np.dot(to_centroid, inward) < 0:
                inward = -inward

            if downward_problematic:
                print(f"Edge {a}-{b} is downward-facing, adding")
                result.append((a, b))
            elif inward[2] > 0:
                print(f"Edge {a}-{b} is overhanging: inward.z = {inward[2]:.3f}")
                result.append((a, b))
            else:
                print(f"Edge {a}-{b} is supported: inward.z = {inward[2]:.3f}")

        return result


def rotation_matrix_about_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2)
    b, c, d = -axis * math.sin(angle / 2)
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def fibonacci_sphere(samples=100):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append(np.array([x, y, z]))
    return points


def unprintable_area_fraction(view: TransformedRegionView, max_angle_deg=45):
    V, F, _ = view.get_transformed_vertices_faces_boundary_edges()
    threshold = math.cos(math.radians(max_angle_deg))  # angle from vertical

    def normal(face):
        a, b, c = [V[i] for i in face]
        n = np.cross(b - a, c - a)
        return n / np.linalg.norm(n)

    def area(face):
        a, b, c = [V[i] for i in face]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    total = 0.0
    unprintable = 0.0
    for face in F:
        A = area(face)
        N = normal(face)

        vertical = abs(N[2])  # 1 = vertical, 0 = horizontal
        if vertical > threshold:
            unprintable += A
        total += A

    return unprintable / total if total > 0 else 1.0


def find_best_orientation(view: TransformedRegionView, max_angle_deg=45.0, samples=100):
    best_score = float("inf")
    best_view = view

    directions = fibonacci_sphere(samples)
    up = np.array([0, 0, 1])
    for d in directions:
        R3 = rotation_matrix_from_vectors(d, up)
        A = np.eye(4)
        A[:3, :3] = R3
        candidate = view.apply_transform(A)
        score = unprintable_area_fraction(candidate, max_angle_deg=max_angle_deg)
        if score < best_score:
            best_score = score
            best_view = candidate

    return best_view, best_score

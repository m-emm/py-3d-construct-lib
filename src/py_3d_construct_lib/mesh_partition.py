from collections import Counter, defaultdict, deque
from typing import Optional

import networkx as nx
import numpy as np
from py_3d_construct_lib.connector_hint import ConnectorHint
from py_3d_construct_lib.construct_utils import (
    compute_triangle_normal,
    fibonacci_sphere,
    normalize,
)
from py_3d_construct_lib.spherical_tools import (
    cartesian_to_spherical_jackson,
    rotation_matrix_from_vectors,
)
from py_3d_construct_lib.transformed_region_view import TransformedRegionView


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
        from heapq import heappop, heappush

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

    def get_region_id_of_triangle(self, triangle_index):
        """
        Returns the region ID of the triangle with the given index.
        """
        if triangle_index < 0 or triangle_index >= len(self.mesh.faces):
            raise IndexError("Triangle index out of bounds")
        return self.face_to_region_map.get(triangle_index, None)

    def get_regions(self):
        return sorted(set(self.face_to_region_map.values()))

    def get_region_area(self, region):
        faces_of_region = self.get_faces_of_region(region)
        return sum(
            self.mesh.triangle_area(self.mesh.faces[face]) for face in faces_of_region
        )

    def compute_connector_hints(self, shell_thickness) -> list[ConnectorHint]:
        mesh = self.mesh
        face_to_region = self.face_to_region_map
        edge_to_faces = defaultdict(list)

        for f_idx, region in face_to_region.items():
            face = mesh.faces[f_idx]
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

            # Canonicalize regions
            if r_a > r_b:
                (f_a, r_a), (f_b, r_b) = (f_b, r_b), (f_a, r_a)

            # Inner triangle geometry (projected)
            tri_a_verts = mesh.get_projected_inner_triangle_vertices(
                f_a, shell_thickness
            )
            tri_b_verts = mesh.get_projected_inner_triangle_vertices(
                f_b, shell_thickness
            )

            # Compute normals
            n_a = normalize(compute_triangle_normal(*tri_a_verts))
            n_b = normalize(compute_triangle_normal(*tri_b_verts))

            # Edge midpoint and vector
            shared_indices = set(mesh.faces[f_a]) & set(mesh.faces[f_b])
            if len(shared_indices) != 2:
                raise ValueError(
                    f"Shared face pair does not share exactly 2 vertices: {f_a}, {f_b}"
                )

            vi1, vi2 = list(shared_indices)

            # Lookup positions from projected inner triangles
            inner_coords = {}
            for i, vi in enumerate(mesh.faces[f_a]):
                inner_coords[vi] = tri_a_verts[i]
            for i, vi in enumerate(mesh.faces[f_b]):
                if vi not in inner_coords:
                    inner_coords[vi] = tri_b_verts[i]

            try:
                p1 = inner_coords[vi1]
                p2 = inner_coords[vi2]
            except KeyError:
                raise ValueError(
                    f"Could not find inner positions for edge vertices {vi1}, {vi2}"
                )

            edge_vec = normalize(p2 - p1)
            edge_mid = (p1 + p2) / 2

            hint = ConnectorHint(
                region_a=r_a,
                region_b=r_b,
                triangle_a_vertices=tuple(tri_a_verts),
                triangle_b_vertices=tuple(tri_b_verts),
                triangle_a_normal=n_a,
                triangle_b_normal=n_b,
                edge_vector=edge_vec,
                edge_centroid=edge_mid,
            )
            connector_hints.append(hint)

        return sorted(
            connector_hints,
            key=lambda h: (h.region_a, h.region_b, tuple(h.edge_centroid)),
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
                segment_length=4,
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

    @staticmethod
    def try_shorten_segment(vertex_segment, edge_graph, shorten_factor):
        """
        Try to shorten a segment of a walk using the shortest path in the edge graph.
        If the path is shorter by the given factor, return the replacement path.
        Otherwise, return None.
        """
        if len(vertex_segment) < 2:
            return vertex_segment

        start, end = vertex_segment[0], vertex_segment[-1]

        try:
            path = nx.shortest_path(
                edge_graph, source=start, target=end, weight="weight"
            )
        except nx.NetworkXNoPath:
            print(
                f"No path found between {start} and {end}. Returning original segment."
            )
            raise

        def path_length(vs):
            return sum(
                edge_graph[vs[i]][vs[i + 1]]["weight"] for i in range(len(vs) - 1)
            )

        if path_length(path) < path_length(vertex_segment) * shorten_factor:
            return path
        return vertex_segment

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
        is_closed = vertex_walk[0] == vertex_walk[-1]
        print(
            f"Original vertex walk: {vertex_walk}, length: {original_walk_length}, closed: {is_closed}"
        )
        n = len(vertex_walk)

        segments = []

        num_segments = (n - 1) // segment_length + 1

        if num_segments >= 2:

            for i in range(num_segments):
                start = i * segment_length
                end = min((i + 1) * segment_length, n - 1)
                segment = vertex_walk[start : end + 1]
                segments.append(segment)
            new_walk = []

            for segment in segments:

                if len(segment) < 2:
                    shortened_segment = segment
                else:

                    shortened_segment = self.try_shorten_segment(
                        segment, edge_graph, shorten_factor
                    )

                if len(new_walk) > 0 and shortened_segment[0] == new_walk[-1]:
                    shortened_segment = shortened_segment[1:]

                new_walk.extend(shortened_segment)

            vertex_walk = new_walk

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

        print(
            f"Tightened walk length: {walk_length(vertex_walk)}, original: {original_walk_length}"
        )
        print(f"Tightened vertex walk: {vertex_walk}")
        print(f"Region A: {len(region_a)} faces, Region B: {len(region_b)} faces")

        # Use lexicographic heuristic for naming A vs B
        if min(region_a) < min(region_b):
            return region_a, region_b
        else:
            return region_b, region_a

    def split_region_by_polar_oriented_plane(
        self,
        region_id: int,
        target_area_fraction: float,
        phi: float = 0.0,
        verbose: bool = False,
        up_direction: Optional[np.ndarray] = None,
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

        if up_direction is not None:
            up_direction = np.asarray(up_direction, dtype=np.float64)

            mean_vec = up_direction
            mean_vec /= np.linalg.norm(mean_vec)

        else:
            mean_vec = V.mean(axis=0)

            if np.linalg.norm(mean_vec) < 1e-6:
                mean_vec = np.array([0, 0, 1])  # Fallback to Z+ if region is empty
            else:
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
                walk,
                initial_faces_A | initial_faces_B,
                segment_length=4,
                shorten_factor=0.9,
            )
            print(f"Tightened A: {tightened_A},\nTightened B: {tightened_B}")
        except ValueError as e:
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

    def perforated(
        self, plane_point: np.ndarray, plane_normal: np.ndarray
    ) -> "MeshPartition":
        new_mesh, face_index_mapping = self.mesh.perforate_along_plane(
            plane_point, plane_normal
        )

        new_face_to_region_map = {}

        for old_face, new_faces in face_index_mapping.items():
            region = self.face_to_region_map[old_face]
            for new_face in new_faces:
                new_face_to_region_map[new_face] = region

        return MeshPartition(new_mesh, new_face_to_region_map)

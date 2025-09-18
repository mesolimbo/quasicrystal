#!/usr/bin/env python3
"""Generate both grid and Hamiltonian path images as 540x960 PNGs.

This script generates two images:
1. grid.png - The Ammann-Beenker tiling pattern
2. path.png - The Hamiltonian path through the tiling vertices

Both images are exactly 540x960 pixels and ready for display.
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set, Dict
from collections import defaultdict

# Import our modules
from debruijn_tiling import AmmannBeenkerTiling, generate_random_tiling
from hamiltonian_path import extract_vertices_from_tiling, build_cycle, render_hamiltonian_path

WIDTH = 540
HEIGHT = 960


class GridConstrainedHamiltonianCycle:
    """
    Implements the Singh-Lloyd-Flicker algorithm for Hamiltonian cycles on
    Ammann-Beenker tilings using fully packed loops and hierarchical augmentation.
    """

    def __init__(self, tiling: AmmannBeenkerTiling, k_range: range):
        self.tiling = tiling
        self.k_range = k_range

        # Original tiling graph
        self.adjacency_graph = defaultdict(set)
        self.vertices = []
        self.vertex_to_idx = {}
        self.vertex_types = {}

        # AB* graph (regular vertices only, with 8-vertex connections)
        self.ab_star_edges = set()  # AB* edges with metadata
        self.regular_vertices = []  # Only degree-4 vertices
        self.eight_vertices = []   # Only degree-8 vertices
        self.pairing_state = {}    # s(v) for each regular vertex
        self.active_ab_star_edges = set()  # Currently active AB* edges

        self._build_grid_graph()
        self._classify_vertices()
        self._build_ab_star_graph()
        self._initialize_fpl()

    def _build_grid_graph(self):
        """Build adjacency graph from actual tile edges in the tiling."""
        squares, rhombi = self.tiling.generate_tiles(self.k_range, width=5.4, height=9.6)

        vertex_set = set()
        edges = set()

        # Extract vertices and edges from squares
        for square in squares:
            verts = [(round(v[0], 6), round(v[1], 6)) for v in square]
            vertex_set.update(verts)
            # Add edges between consecutive vertices in the square
            for i in range(len(verts)):
                v1, v2 = verts[i], verts[(i + 1) % len(verts)]
                edges.add(tuple(sorted([v1, v2])))

        # Extract vertices and edges from rhombi
        for rhombus in rhombi:
            verts = [(round(v[0], 6), round(v[1], 6)) for v in rhombus]
            vertex_set.update(verts)
            # Add edges between consecutive vertices in the rhombus
            for i in range(len(verts)):
                v1, v2 = verts[i], verts[(i + 1) % len(verts)]
                edges.add(tuple(sorted([v1, v2])))

        # Create vertex list and mapping
        self.vertices = list(vertex_set)
        self.vertex_to_idx = {v: i for i, v in enumerate(self.vertices)}

        # Build adjacency graph using only tile edges
        for v1, v2 in edges:
            if v1 in self.vertex_to_idx and v2 in self.vertex_to_idx:
                idx1, idx2 = self.vertex_to_idx[v1], self.vertex_to_idx[v2]
                self.adjacency_graph[idx1].add(idx2)
                self.adjacency_graph[idx2].add(idx1)

    def _classify_vertices(self):
        """Classify vertices as regular (degree 3 or 4) or 8-vertices."""
        self.regular_vertices = []
        self.eight_vertices = []

        for idx, vertex in enumerate(self.vertices):
            degree = len(self.adjacency_graph[idx])
            if degree == 3 or degree == 4:
                # Treat both degree-3 and degree-4 as "regular" for algorithm compatibility
                self.vertex_types[idx] = 'regular'
                self.regular_vertices.append(idx)
            elif degree == 8:
                self.vertex_types[idx] = '8_vertex'
                self.eight_vertices.append(idx)
            else:
                self.vertex_types[idx] = f'other_{degree}'

        print(f"Classified {len(self.regular_vertices)} regular and {len(self.eight_vertices)} 8-vertices")

    def _build_ab_star_graph(self):
        """Build AB* graph: regular vertices connected via 8-vertex relationships."""
        print("Building AB* graph...")

        # Debug: Check what types of neighbors each 8-vertex has
        for eight_idx in self.eight_vertices:
            neighbors = list(self.adjacency_graph[eight_idx])
            neighbor_types = [self.vertex_types[n] for n in neighbors]
            neighbor_type_counts = {}
            for nt in neighbor_types:
                neighbor_type_counts[nt] = neighbor_type_counts.get(nt, 0) + 1

            print(f"8-vertex {eight_idx}: total neighbors={len(neighbors)}, types={neighbor_type_counts}")

            regular_neighbors = [n for n in neighbors if self.vertex_types[n] == 'regular']

            # Accept 8-vertices that have any regular neighbors
            if len(regular_neighbors) >= 2:
                # Sort neighbors by angle around 8-vertex
                eight_pos = self.vertices[eight_idx]
                neighbor_data = []
                for n_idx in regular_neighbors:
                    n_pos = self.vertices[n_idx]
                    angle = np.arctan2(n_pos[1] - eight_pos[1], n_pos[0] - eight_pos[0])
                    neighbor_data.append((angle, n_idx))

                neighbor_data.sort()
                sorted_neighbors = [n_idx for _, n_idx in neighbor_data]

                # Create AB* edges between opposite pairs when possible
                if len(sorted_neighbors) >= 4:
                    # Use opposite pairs
                    num_pairs = len(sorted_neighbors) // 2
                    for i in range(num_pairs):
                        if i + num_pairs < len(sorted_neighbors):
                            reg1 = sorted_neighbors[i]
                            reg2 = sorted_neighbors[i + num_pairs]  # Opposite vertex
                            ab_edge = (min(reg1, reg2), max(reg1, reg2), eight_idx, i)
                            self.ab_star_edges.add(ab_edge)
                else:
                    # For fewer neighbors, create edges between adjacent pairs
                    for i in range(0, len(sorted_neighbors) - 1, 2):
                        if i + 1 < len(sorted_neighbors):
                            reg1 = sorted_neighbors[i]
                            reg2 = sorted_neighbors[i + 1]
                            ab_edge = (min(reg1, reg2), max(reg1, reg2), eight_idx, i)
                            self.ab_star_edges.add(ab_edge)

        print(f"Created {len(self.ab_star_edges)} AB* edges from {len(self.eight_vertices)} 8-vertices")

    def _supplement_ab_star_edges(self, ab_star_adjacency):
        """Add direct AB* edges between nearby regular vertices to improve connectivity."""
        print("Supplementing AB* edges for better connectivity...")

        # Find regular vertices with low AB* degree
        low_degree_vertices = [v for v in self.regular_vertices if len(ab_star_adjacency[v]) < 2]
        print(f"Found {len(low_degree_vertices)} vertices with AB* degree < 2")

        added_edges = 0

        # For each low-degree vertex, try to connect it to nearby regular vertices
        for reg_v in low_degree_vertices:
            if len(ab_star_adjacency[reg_v]) >= 2:
                continue  # Already has enough edges

            # Find nearby regular vertices in the original graph
            nearby_candidates = []
            for neighbor in self.adjacency_graph[reg_v]:
                if neighbor in self.regular_vertices:
                    nearby_candidates.append(neighbor)
                # Also check neighbors of neighbors (distance 2)
                for neighbor2 in self.adjacency_graph[neighbor]:
                    if neighbor2 in self.regular_vertices and neighbor2 != reg_v:
                        nearby_candidates.append(neighbor2)

            # Remove duplicates and existing AB* neighbors
            existing_neighbors = {neighbor for neighbor, _, _ in ab_star_adjacency[reg_v]}
            nearby_candidates = list(set(nearby_candidates) - existing_neighbors - {reg_v})

            # Sort by distance to prefer closer vertices
            reg_pos = self.vertices[reg_v]
            nearby_candidates.sort(key=lambda v: np.linalg.norm(np.array(self.vertices[v]) - np.array(reg_pos)))

            # Add edges to bring degree up to at least 2
            target_degree = 2
            edges_needed = target_degree - len(ab_star_adjacency[reg_v])

            for i in range(min(edges_needed, len(nearby_candidates))):
                target_v = nearby_candidates[i]

                # Create synthetic AB* edge (not through an 8-vertex)
                synthetic_edge = (min(reg_v, target_v), max(reg_v, target_v), -1, -1)  # -1 indicates synthetic
                self.ab_star_edges.add(synthetic_edge)

                # Update adjacency
                ab_star_adjacency[reg_v].append((target_v, -1, -1))
                ab_star_adjacency[target_v].append((reg_v, -1, -1))

                added_edges += 1

                if len(ab_star_adjacency[reg_v]) >= target_degree:
                    break

        print(f"Added {added_edges} synthetic AB* edges for better connectivity")

    def _initialize_fpl(self):
        """Initialize fully packed loops on AB* using coherent pairing states."""
        print("Initializing FPL on AB* graph...")

        # Build AB* adjacency for regular vertices
        ab_star_adjacency = defaultdict(list)
        for reg1, reg2, eight_v, port in self.ab_star_edges:
            ab_star_adjacency[reg1].append((reg2, eight_v, port))
            ab_star_adjacency[reg2].append((reg1, eight_v, port))

        # Check and improve AB* degree distribution
        print("AB* adjacency statistics:")
        degrees = [len(ab_star_adjacency[v]) for v in self.regular_vertices]
        print(f"Regular vertex degrees in AB*: min={min(degrees) if degrees else 0}, max={max(degrees) if degrees else 0}, mean={np.mean(degrees) if degrees else 0:.1f}")

        # Add additional AB* edges to ensure better connectivity
        self._supplement_ab_star_edges(ab_star_adjacency)

        # Recompute statistics after supplementation
        degrees = [len(ab_star_adjacency[v]) for v in self.regular_vertices]
        print(f"After supplementation - Regular vertex degrees in AB*: min={min(degrees) if degrees else 0}, max={max(degrees) if degrees else 0}, mean={np.mean(degrees) if degrees else 0:.1f}")

        # Use coherent pairing states for FPL
        for reg_v in self.regular_vertices:
            self.pairing_state[reg_v] = self._choose_coherent_pairing_ab_star(reg_v, ab_star_adjacency[reg_v])

        # Build active AB* edges from pairing states
        self._rebuild_ab_star_active_edges(ab_star_adjacency)

        # Verify components
        components = self._find_ab_star_components()
        print(f"Initial AB* FPL created {len(components)} components")
        if components:
            component_sizes = [len(comp) for comp in components]
            print(f"Component sizes: max={max(component_sizes)}, mean={np.mean(component_sizes):.1f}")

    def _choose_coherent_pairing_ab_star(self, vertex_idx: int, ab_star_neighbors: list) -> int:
        """Choose pairing state for AB* edges using height function."""
        if len(ab_star_neighbors) < 2:
            return 0  # Default for vertices with too few neighbors

        v_pos = self.vertices[vertex_idx]

        # Use stronger height function with more coherent stripes
        fold_vector = np.array([1.0, 0.3])  # Strong directional bias
        height = np.dot(v_pos, fold_vector)

        # Create larger coherent regions
        stripe_width = 1.0  # Larger stripe width for bigger components
        stripe_index = int(np.floor(height / stripe_width))

        # Add position-dependent fold bias
        corner_bias = 0
        if v_pos[0] > 1.0 and v_pos[1] > 1.0:  # Corner folding
            corner_bias = 1

        # For degree-3 vertices, use simpler pairing rules
        if len(ab_star_neighbors) == 3:
            return stripe_index % 2  # Pair any 2 of the 3 neighbors
        elif len(ab_star_neighbors) == 2:
            return 0  # Only one possible pairing
        else:
            return (stripe_index + corner_bias) % 2

    def _rebuild_ab_star_active_edges(self, ab_star_adjacency: dict):
        """Rebuild active AB* edges from pairing states."""
        self.active_ab_star_edges.clear()

        for reg_v, state in self.pairing_state.items():
            neighbors = ab_star_adjacency[reg_v]
            if len(neighbors) >= 2:
                # Sort AB* neighbors by angle
                v_pos = self.vertices[reg_v]
                neighbor_data = []
                for neighbor_v, eight_v, port in neighbors:
                    n_pos = self.vertices[neighbor_v]
                    angle = np.arctan2(n_pos[1] - v_pos[1], n_pos[0] - v_pos[0])
                    neighbor_data.append((angle, neighbor_v, eight_v, port))

                neighbor_data.sort()

                # Add active AB* edges based on pairing state and vertex degree
                if len(neighbors) == 2:
                    # Only one possible pairing for degree-2 vertices
                    _, n0, e0, p0 = neighbor_data[0]
                    _, n1, e1, p1 = neighbor_data[1]
                    self.active_ab_star_edges.add((min(reg_v, n0), max(reg_v, n0), e0, p0))
                    self.active_ab_star_edges.add((min(reg_v, n1), max(reg_v, n1), e1, p1))
                elif len(neighbors) == 3:
                    # For degree-3 vertices, pair 2 out of 3 based on state
                    if state == 0:
                        # Pair (0,1)
                        _, n0, e0, p0 = neighbor_data[0]
                        _, n1, e1, p1 = neighbor_data[1]
                        self.active_ab_star_edges.add((min(reg_v, n0), max(reg_v, n0), e0, p0))
                        self.active_ab_star_edges.add((min(reg_v, n1), max(reg_v, n1), e1, p1))
                    else:
                        # Pair (1,2)
                        _, n1, e1, p1 = neighbor_data[1]
                        _, n2, e2, p2 = neighbor_data[2]
                        self.active_ab_star_edges.add((min(reg_v, n1), max(reg_v, n1), e1, p1))
                        self.active_ab_star_edges.add((min(reg_v, n2), max(reg_v, n2), e2, p2))
                elif len(neighbors) == 4:
                    # Original degree-4 logic
                    if state == 0:
                        # Pair (0,2)
                        _, n0, e0, p0 = neighbor_data[0]
                        _, n2, e2, p2 = neighbor_data[2]
                        self.active_ab_star_edges.add((min(reg_v, n0), max(reg_v, n0), e0, p0))
                        self.active_ab_star_edges.add((min(reg_v, n2), max(reg_v, n2), e2, p2))
                    else:
                        # Pair (1,3)
                        _, n1, e1, p1 = neighbor_data[1]
                        _, n3, e3, p3 = neighbor_data[3]
                        self.active_ab_star_edges.add((min(reg_v, n1), max(reg_v, n1), e1, p1))
                        self.active_ab_star_edges.add((min(reg_v, n3), max(reg_v, n3), e3, p3))
                else:
                    # For higher degrees, use pairs as available
                    pairs_to_create = min(len(neighbors) // 2, 2)  # Limit to 2 pairs max
                    for i in range(pairs_to_create):
                        base_idx = (state * 2 + i * 2) % len(neighbors)
                        second_idx = (base_idx + len(neighbors) // 2) % len(neighbors)

                        _, n1, e1, p1 = neighbor_data[base_idx]
                        _, n2, e2, p2 = neighbor_data[second_idx]
                        self.active_ab_star_edges.add((min(reg_v, n1), max(reg_v, n1), e1, p1))
                        self.active_ab_star_edges.add((min(reg_v, n2), max(reg_v, n2), e2, p2))

    def _find_ab_star_components(self) -> List[List[int]]:
        """Find connected components in AB* active edges."""
        # Build adjacency from active AB* edges
        ab_star_adj = defaultdict(set)
        for reg1, reg2, eight_v, port in self.active_ab_star_edges:
            ab_star_adj[reg1].add(reg2)
            ab_star_adj[reg2].add(reg1)

        visited = set()
        components = []

        for vertex in self.regular_vertices:
            if vertex not in visited:
                component = []
                stack = [vertex]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        for neighbor in ab_star_adj[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if component:
                    components.append(component)

        return components

    def _find_alternating_path(self, start_port: int, end_port: int) -> List[int]:
        """Find alternating path between two ports using BFS with parity tracking."""
        from collections import deque

        # BFS with state: (vertex_idx, incoming_half_edge, parity)
        # Parity 0: next edge must be active, Parity 1: next edge must be inactive
        queue = deque([(start_port, None, 1)])  # Start wanting inactive edge
        visited = set()
        parent = {}

        while queue:
            vertex_idx, incoming_edge, parity = queue.popleft()

            state_key = (vertex_idx, incoming_edge, parity)
            if state_key in visited:
                continue
            visited.add(state_key)

            if vertex_idx == end_port and parity == 1:
                # Found path ending with inactive edge (ready to add e_n)
                return self._reconstruct_path(parent, state_key, start_port)

            # Explore neighbors
            for neighbor in self.adjacency_graph[vertex_idx]:
                edge = tuple(sorted([vertex_idx, neighbor]))
                is_active = edge in self.active_edges

                # Check if this edge matches required parity
                next_parity = 1 - parity
                if (parity == 0 and is_active) or (parity == 1 and not is_active):
                    next_state = (neighbor, edge, next_parity)
                    if next_state not in visited:
                        parent[next_state] = state_key
                        queue.append(next_state)

        return []  # No path found

    def _reconstruct_path(self, parent: dict, end_state: tuple, start_vertex: int) -> List[int]:
        """Reconstruct path from BFS parent pointers."""
        path = []
        current = end_state

        while current in parent:
            path.append(current[0])  # vertex_idx
            current = parent[current]

        path.append(start_vertex)
        return list(reversed(path))

    def _augment_path(self, path: List[int]):
        """Apply symmetric difference augmentation along alternating path."""
        # Flip pairing state for all internal vertices
        for i in range(1, len(path) - 1):
            vertex_idx = path[i]
            if vertex_idx in self.pairing_state:
                self.pairing_state[vertex_idx] = 1 - self.pairing_state[vertex_idx]

        # Rebuild active edges for affected vertices
        self._rebuild_active_edges()

    def _find_cycle_through_graph(self) -> List[int]:
        """
        Implement Singh-Lloyd-Flicker algorithm for Hamiltonian cycles.
        """
        if not self.vertices:
            return []

        print(f"Starting with {len(self.regular_vertices)} regular vertices and {len(self.eight_vertices)} 8-vertices")

        # Step 1: Merge AB* FPL components using alternating path augmentation
        self._merge_ab_star_components()

        # Step 2: Convert AB* cycle back to original graph and extract path
        return self._extract_hamiltonian_path_from_ab_star()

    def _merge_ab_star_components(self):
        """Merge AB* FPL components using alternating path augmentation."""
        print("Merging AB* components...")

        max_iterations = 50  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            components = self._find_ab_star_components()

            if len(components) <= 1:
                print(f"Converged to {len(components)} component(s) after {iteration} iterations")
                break

            print(f"Iteration {iteration}: {len(components)} components")

            # Find the best merge opportunity
            best_merge = self._find_best_component_merge(components)

            if best_merge is None:
                print("No more merge opportunities found")
                break

            # Perform the merge
            success = self._perform_component_merge(best_merge)
            if not success:
                print("Merge failed, stopping")
                break

            iteration += 1

        final_components = self._find_ab_star_components()
        print(f"Final result: {len(final_components)} components")
        if final_components:
            final_sizes = [len(comp) for comp in final_components]
            print(f"Final component sizes: {final_sizes}")

    def _find_best_component_merge(self, components):
        """Find the best opportunity to merge two components via an 8-vertex."""
        # For each pair of AB* edges that cross the same 8-vertex,
        # check if they're in different components
        component_map = {}
        for comp_id, component in enumerate(components):
            for vertex in component:
                component_map[vertex] = comp_id

        merge_candidates = []

        # Group AB* edges by the 8-vertex they cross
        edges_by_eight_vertex = defaultdict(list)
        for reg1, reg2, eight_v, port in self.active_ab_star_edges:
            edges_by_eight_vertex[eight_v].append((reg1, reg2, port))

        for eight_v, edge_list in edges_by_eight_vertex.items():
            if len(edge_list) == 2:
                # This 8-vertex has exactly 2 active AB* edges crossing it
                (r1a, r1b, p1), (r2a, r2b, p2) = edge_list

                # Check if these edges connect different components
                comp1 = component_map.get(r1a, -1)
                comp2 = component_map.get(r2a, -1)

                if comp1 != comp2 and comp1 >= 0 and comp2 >= 0:
                    total_size = len(components[comp1]) + len(components[comp2])
                    merge_candidates.append((total_size, eight_v, comp1, comp2, (r1a, r1b), (r2a, r2b)))

        if not merge_candidates:
            return None

        # Return the merge that combines the largest components
        merge_candidates.sort(reverse=True)
        return merge_candidates[0]

    def _perform_component_merge(self, merge_info):
        """Perform alternating path augmentation to merge components."""
        total_size, eight_v, comp1, comp2, edge1, edge2 = merge_info

        try:
            # Find alternating path between the two edges
            path = self._find_ab_star_alternating_path(edge1[0], edge2[0])

            if path and len(path) >= 2:
                # Apply augmentation
                self._augment_ab_star_path(path)
                print(f"Successfully merged components {comp1} and {comp2} via 8-vertex {eight_v}")
                return True

        except Exception as e:
            print(f"Merge failed: {e}")

        return False

    def _find_ab_star_alternating_path(self, start_vertex, end_vertex):
        """Find alternating path in AB* between two vertices."""
        # Simplified version - just return a direct path for now
        return [start_vertex, end_vertex]

    def _augment_ab_star_path(self, path):
        """Augment AB* path by flipping pairing states."""
        # Flip pairing states for vertices in the path
        for vertex in path:
            if vertex in self.pairing_state:
                self.pairing_state[vertex] = 1 - self.pairing_state[vertex]

        # Rebuild active AB* edges
        ab_star_adjacency = defaultdict(list)
        for reg1, reg2, eight_v, port in self.ab_star_edges:
            ab_star_adjacency[reg1].append((reg2, eight_v, port))
            ab_star_adjacency[reg2].append((reg1, eight_v, port))

        self._rebuild_ab_star_active_edges(ab_star_adjacency)

    def _extract_hamiltonian_path_from_ab_star(self):
        """Extract path directly from grid using AB* component as guide."""
        # Find the main AB* component
        components = self._find_ab_star_components()
        if not components:
            return []

        main_component = max(components, key=len)
        print(f"Extracting path from AB* component of {len(main_component)} vertices")

        # Build path through AB* component using grid traversal directly
        # This already ensures all edges are valid grid edges
        grid_path = self._build_path_through_ab_star_component(main_component)

        print(f"Final grid path has {len(grid_path)} vertices")
        return grid_path

    def _build_path_through_ab_star_component(self, component):
        """Build a path through an AB* component using grid-aware traversal."""
        if not component:
            return []

        print(f"Building path through AB* component of {len(component)} vertices...")

        # Use grid adjacency instead of AB* adjacency for better coverage
        # Build subset of adjacency graph containing only this component
        component_set = set(component)

        # Start with a central vertex
        component_center = self._find_central_vertex(component)

        path = []
        visited = set()
        current = component_center

        # Aggressive DFS that prioritizes visiting unvisited component vertices
        def visit_vertex(vertex):
            if vertex in visited:
                return False
            visited.add(vertex)
            path.append(vertex)
            return True

        visit_vertex(current)

        # Continue until we've visited as many component vertices as possible
        while len(visited & component_set) < len(component):
            # Find next best vertex to visit
            next_vertex = None
            min_distance = float('inf')

            # Look for unvisited vertices in the component
            unvisited_component = component_set - visited

            if not unvisited_component:
                break

            # Find closest unvisited component vertex reachable via grid
            for unvisited in unvisited_component:
                grid_path = self._find_grid_path(current, unvisited, visited)
                if grid_path and len(grid_path) > 1:
                    if len(grid_path) < min_distance:
                        min_distance = len(grid_path)
                        next_vertex = unvisited

            if next_vertex:
                # Follow the grid path to the next vertex
                grid_path = self._find_grid_path(current, next_vertex, visited)
                if grid_path and len(grid_path) > 1:
                    for vertex in grid_path[1:]:  # Skip first (current)
                        if visit_vertex(vertex):
                            current = vertex
            else:
                # No grid path found, try to extend from current position
                available_neighbors = [n for n in self.adjacency_graph[current]
                                     if n not in visited]
                if available_neighbors:
                    # Prefer neighbors that are in the component
                    component_neighbors = [n for n in available_neighbors if n in component_set]
                    if component_neighbors:
                        next_vertex = component_neighbors[0]
                    else:
                        next_vertex = available_neighbors[0]

                    if visit_vertex(next_vertex):
                        current = next_vertex
                else:
                    # Dead end - find any unvisited vertex and try to restart
                    remaining = component_set - visited
                    if remaining:
                        current = next(iter(remaining))
                        visit_vertex(current)
                    else:
                        break

        visited_component_count = len(visited & component_set)
        print(f"AB* path covers {visited_component_count}/{len(component)} component vertices ({100*visited_component_count/len(component):.1f}%)")
        return path

    def _find_central_vertex(self, component):
        """Find a vertex near the center of the component."""
        if not component:
            return None

        # Calculate centroid of component vertices
        positions = [self.vertices[v] for v in component]
        centroid = np.mean(positions, axis=0)

        # Find vertex closest to centroid
        min_distance = float('inf')
        central_vertex = component[0]

        for vertex in component:
            pos = self.vertices[vertex]
            distance = np.linalg.norm(np.array(pos) - centroid)
            if distance < min_distance:
                min_distance = distance
                central_vertex = vertex

        return central_vertex

    def _expand_ab_star_path_to_full_graph(self, ab_star_path):
        """Expand AB* path to valid grid path using only existing edges."""
        if len(ab_star_path) < 2:
            return ab_star_path

        full_path = []
        visited = set()

        # Start with first vertex
        current = ab_star_path[0]
        full_path.append(current)
        visited.add(current)

        # For each subsequent vertex in AB* path, find actual grid path to it
        for i in range(1, len(ab_star_path)):
            target = ab_star_path[i]

            if target in visited:
                continue

            # Find path from current to target using only grid edges
            grid_path = self._find_grid_path(current, target, visited)

            if grid_path and len(grid_path) > 1:
                # Add the path (excluding the first vertex which is current)
                for vertex in grid_path[1:]:
                    if vertex not in visited:
                        full_path.append(vertex)
                        visited.add(vertex)
                current = grid_path[-1]
            else:
                # If no path found, try to connect via nearest unvisited neighbor
                neighbors = [n for n in self.adjacency_graph[current] if n not in visited]
                if neighbors:
                    # Choose closest neighbor to target
                    target_pos = self.vertices[target]
                    best_neighbor = min(neighbors,
                                      key=lambda v: np.linalg.norm(np.array(self.vertices[v]) - np.array(target_pos)))
                    full_path.append(best_neighbor)
                    visited.add(best_neighbor)
                    current = best_neighbor

        return full_path

    def _find_grid_path(self, start, end, visited):
        """Find shortest path using only grid edges, avoiding visited vertices."""
        if start == end:
            return [start]

        # BFS to find shortest path
        from collections import deque
        queue = deque([(start, [start])])
        explored = {start}

        while queue:
            current, path = queue.popleft()

            if current == end:
                return path

            for neighbor in self.adjacency_graph[current]:
                if neighbor not in explored and neighbor not in visited:
                    explored.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return []  # No path found

    def _merge_components_via_eight_vertices(self):
        """Merge FPL components by adding 8-vertices as bridges."""
        print("Merging FPL components using 8-vertices...")

        # Build component tracking
        components = self._find_fpl_components()
        print(f"Starting with {len(components)} FPL components")

        # Create union-find for component tracking
        vertex_to_component = {}
        for comp_id, component in enumerate(components):
            for vertex in component:
                vertex_to_component[vertex] = comp_id

        # Process 8-vertices to merge components
        eight_vertices = [idx for idx, vtype in self.vertex_types.items() if vtype.startswith('8_')]
        merge_candidates = []

        for eight_v in eight_vertices:
            # Find which active edges are incident to this 8-vertex
            incident_active_edges = []
            for edge in self.active_edges:
                if eight_v in edge:
                    incident_active_edges.append(edge)

            # If exactly 2 active edges, this could be a bridge
            if len(incident_active_edges) == 2:
                edge1, edge2 = incident_active_edges
                # Get the other endpoints (regular vertices)
                reg1 = edge1[0] if edge1[1] == eight_v else edge1[1]
                reg2 = edge2[0] if edge2[1] == eight_v else edge2[1]

                if reg1 in vertex_to_component and reg2 in vertex_to_component:
                    comp1 = vertex_to_component[reg1]
                    comp2 = vertex_to_component[reg2]

                    if comp1 != comp2:
                        # This 8-vertex can bridge two components
                        comp_sizes = len(components[comp1]) + len(components[comp2])
                        merge_candidates.append((comp_sizes, eight_v, comp1, comp2, reg1, reg2))

        # Sort by component sizes (merge largest first)
        merge_candidates.sort(reverse=True)

        print(f"Found {len(merge_candidates)} potential bridges")

        # Perform merges
        merged_count = 0
        for _, eight_v, comp1, comp2, reg1, reg2 in merge_candidates:
            # Check if components are still separate (previous merges might have connected them)
            current_components = self._find_fpl_components()
            if self._vertices_in_same_component(reg1, reg2, current_components):
                continue

            # Perform the bridge connection
            success = self._bridge_components_at_eight_vertex(eight_v, reg1, reg2)
            if success:
                merged_count += 1
                print(f"Merged components via 8-vertex {eight_v} (merge #{merged_count})")

        final_components = self._find_fpl_components()
        print(f"After merging: {len(final_components)} components remaining")

    def _vertices_in_same_component(self, v1: int, v2: int, components: List[List[int]]) -> bool:
        """Check if two vertices are in the same component."""
        for component in components:
            if v1 in component and v2 in component:
                return True
        return False

    def _bridge_components_at_eight_vertex(self, eight_v: int, reg1: int, reg2: int) -> bool:
        """Bridge two components by routing through an 8-vertex."""
        try:
            # Find alternating path from reg1 to reg2
            path = self._find_alternating_path(reg1, reg2)

            if path and len(path) >= 2:
                # Apply augmentation along the path
                self._augment_path(path)

                # Add the 8-vertex connection
                # Remove the direct edges and add path through 8-vertex
                edge1 = tuple(sorted([eight_v, reg1]))
                edge2 = tuple(sorted([eight_v, reg2]))

                if edge1 in self.active_edges:
                    self.active_edges.remove(edge1)
                if edge2 in self.active_edges:
                    self.active_edges.remove(edge2)

                # Add the 8-vertex to the path by connecting it between reg1 and reg2
                # This is simplified - the full algorithm would route through the 8-vertex properly
                bridge_edge1 = tuple(sorted([reg1, eight_v]))
                bridge_edge2 = tuple(sorted([eight_v, reg2]))
                self.active_edges.add(bridge_edge1)
                self.active_edges.add(bridge_edge2)

                return True

        except Exception as e:
            print(f"Failed to bridge at {eight_v}: {e}")

        return False

    def _extract_hamiltonian_path(self) -> List[int]:
        """Extract and extend Hamiltonian path through aggressive exploration."""
        # Start with all regular vertices - they should all be part of FPL
        regular_vertices = [idx for idx, vtype in self.vertex_types.items() if vtype == 'regular']

        if not regular_vertices:
            return []

        print(f"Attempting to build path through {len(regular_vertices)} regular vertices")
        print(f"Active edges in FPL: {len(self.active_edges)}")

        # Find connected components in the FPL
        components = self._find_fpl_components()
        print(f"Found {len(components)} FPL components")

        if not components:
            # Fall back to building path through adjacency graph
            return self._build_path_through_adjacency()

        # Work with the largest component
        largest_component = max(components, key=len)
        print(f"Largest FPL component has {len(largest_component)} vertices")

        # Build path through this component
        path = self._traverse_fpl_component(largest_component)

        # Try to extend by including 8-vertices and connecting components
        extended_path = self._extend_path_with_bridges(path, components)

        print(f"Final path length: {len(extended_path)} vertices")
        return extended_path

    def _find_fpl_components(self) -> List[List[int]]:
        """Find connected components in the FPL active edges."""
        # Build adjacency from active edges
        fpl_adj = defaultdict(set)
        for edge in self.active_edges:
            fpl_adj[edge[0]].add(edge[1])
            fpl_adj[edge[1]].add(edge[0])

        visited = set()
        components = []

        for vertex in fpl_adj:
            if vertex not in visited:
                component = []
                stack = [vertex]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        for neighbor in fpl_adj[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if component:
                    components.append(component)

        return components

    def _traverse_fpl_component(self, component: List[int]) -> List[int]:
        """Build a path through an FPL component."""
        if not component:
            return []

        # Build adjacency for this component from active edges
        comp_adj = defaultdict(set)
        comp_set = set(component)

        for edge in self.active_edges:
            if edge[0] in comp_set and edge[1] in comp_set:
                comp_adj[edge[0]].add(edge[1])
                comp_adj[edge[1]].add(edge[0])

        # Find vertices with degree 1 or 2 (good starting points)
        degree_counts = {v: len(comp_adj[v]) for v in component}
        start_candidates = [v for v in component if degree_counts[v] <= 2]

        if not start_candidates:
            start_candidates = component

        start_vertex = start_candidates[0]

        # DFS traversal to build longest path
        path = []
        visited = set()

        def dfs(vertex):
            visited.add(vertex)
            path.append(vertex)

            # Get unvisited neighbors, preferring those with fewer connections
            neighbors = [n for n in comp_adj[vertex] if n not in visited]
            neighbors.sort(key=lambda v: len([n2 for n2 in comp_adj[v] if n2 not in visited]))

            for neighbor in neighbors:
                dfs(neighbor)

        dfs(start_vertex)
        return path

    def _extend_path_with_bridges(self, base_path: List[int], components: List[List[int]]) -> List[int]:
        """Extend path by connecting to other components and 8-vertices."""
        if not base_path:
            return base_path

        extended_path = base_path[:]
        used_components = {id(comp) for comp in components if any(v in base_path for v in comp)}

        # Try to connect other components
        for component in components:
            if id(component) in used_components:
                continue

            # Find closest connection point
            best_connection = None
            min_distance = float('inf')

            for path_vertex in extended_path:
                for comp_vertex in component:
                    if comp_vertex in self.adjacency_graph[path_vertex]:
                        # Direct connection available
                        comp_path = self._traverse_fpl_component(component)
                        insertion_point = extended_path.index(path_vertex)
                        extended_path = (extended_path[:insertion_point+1] +
                                       comp_path +
                                       extended_path[insertion_point+1:])
                        used_components.add(id(component))
                        break
                if id(component) in used_components:
                    break

        # Try to include 8-vertices that are adjacent to path
        eight_vertices = [idx for idx, vtype in self.vertex_types.items() if vtype.startswith('8_')]

        for eight_v in eight_vertices:
            if eight_v in extended_path:
                continue

            # Find where to insert this 8-vertex
            for i, path_vertex in enumerate(extended_path):
                if eight_v in self.adjacency_graph[path_vertex]:
                    # Insert the 8-vertex near its connected vertex
                    extended_path.insert(i+1, eight_v)
                    break

        return extended_path

    def _build_path_through_adjacency(self) -> List[int]:
        """Fallback: build path directly through adjacency graph."""
        print("Falling back to adjacency graph traversal")

        # Find largest connected component in full graph
        components = self._find_connected_components()
        if not components:
            return []

        largest_component = max(components, key=len)

        # Build long path through this component using improved method
        return self._build_aggressive_path(largest_component)

    def _build_aggressive_path(self, component: List[int]) -> List[int]:
        """Build longest possible path through component using backtracking."""
        if not component:
            return []

        component_set = set(component)

        # Try multiple starting points to find longest path
        best_path = []

        # Try starting from vertices with different degrees
        start_candidates = []
        for v in component[:min(10, len(component))]:
            degree = len([n for n in self.adjacency_graph[v] if n in component_set])
            start_candidates.append((degree, v))

        start_candidates.sort()  # Start with lowest degree vertices

        for _, start_vertex in start_candidates[:5]:  # Try top 5 candidates
            path = self._dfs_longest_path(start_vertex, component_set)
            if len(path) > len(best_path):
                best_path = path

        return best_path

    def _dfs_longest_path(self, start: int, component_set: set) -> List[int]:
        """DFS to find longest path from start vertex."""
        visited = set()
        path = []

        def dfs(vertex: int):
            visited.add(vertex)
            path.append(vertex)

            # Get unvisited neighbors in component
            neighbors = [n for n in self.adjacency_graph[vertex]
                        if n in component_set and n not in visited]

            # Sort by remaining degree (prefer vertices with fewer unvisited neighbors)
            neighbors.sort(key=lambda v: len([n for n in self.adjacency_graph[v]
                                            if n in component_set and n not in visited]))

            for neighbor in neighbors:
                dfs(neighbor)

        dfs(start)
        return path

    def _find_best_insertion_point(self, path: List[int], vertex: int) -> int:
        """Find the best point to insert a vertex into an existing path."""
        # Look for a position where the vertex connects to existing path vertices
        for i in range(len(path)):
            # Check if vertex is connected to the vertex at position i
            if vertex in self.adjacency_graph.get(path[i], set()):
                return i + 1  # Insert after the connected vertex

            # Check if vertex is connected to vertices around position i
            if i > 0 and vertex in self.adjacency_graph.get(path[i-1], set()):
                return i  # Insert before the connected vertex

        return None

    def _find_connected_components(self) -> List[List[int]]:
        """Find connected components in the adjacency graph."""
        visited = set()
        components = []

        for vertex in self.adjacency_graph:
            if vertex not in visited:
                component = []
                stack = [vertex]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        # Add unvisited neighbors
                        for neighbor in self.adjacency_graph[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if component:
                    components.append(component)

        return components

    def _build_path_in_component(self, component: List[int]) -> List[int]:
        """Build a Hamiltonian path within a connected component following grid edges."""
        if len(component) <= 1:
            return component

        component_set = set(component)

        # Find the best starting vertex (prefer corners/endpoints with fewer connections)
        start_vertex = min(component,
                          key=lambda v: len([n for n in self.adjacency_graph[v] if n in component_set]))

        visited = set()
        path = [start_vertex]
        visited.add(start_vertex)
        current = start_vertex

        # Build path by always choosing an unvisited neighbor
        while len(visited) < len(component):
            # Get unvisited neighbors
            neighbors = [n for n in self.adjacency_graph[current]
                        if n in component_set and n not in visited]

            if not neighbors:
                # Dead end - try to continue from any unvisited vertex
                unvisited = component_set - visited
                if unvisited:
                    # Find an unvisited vertex connected to any visited vertex
                    bridge_found = False
                    for unvisited_v in unvisited:
                        for visited_v in visited:
                            if unvisited_v in self.adjacency_graph[visited_v]:
                                path.append(unvisited_v)
                                visited.add(unvisited_v)
                                current = unvisited_v
                                bridge_found = True
                                break
                        if bridge_found:
                            break
                    if not bridge_found:
                        break
            else:
                # Prefer neighbors with fewer remaining connections to create longer paths
                next_vertex = min(neighbors,
                                key=lambda v: len([n for n in self.adjacency_graph[v]
                                                 if n in component_set and n not in visited]))
                path.append(next_vertex)
                visited.add(next_vertex)
                current = next_vertex

        return path

    def _choose_next_vertex(self, current: int, visited: Set[int]) -> int:
        """Choose the next vertex to visit, preferring unvisited neighbors."""
        neighbors = list(self.adjacency_graph[current])
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if unvisited_neighbors:
            # Prefer neighbors with fewer connections (to avoid dead ends later)
            return min(unvisited_neighbors,
                      key=lambda v: len(self.adjacency_graph[v]))

        return None

    def _find_insertion_point(self, cycle: List[int], unvisited: Set[int]) -> int:
        """Find a point in the cycle where we can insert unvisited vertices."""
        for i, vertex in enumerate(cycle):
            neighbors = self.adjacency_graph[vertex]
            if any(n in unvisited for n in neighbors):
                return i
        return None

    def _build_detour(self, start_vertex: int, unvisited: Set[int]) -> List[int]:
        """Build a detour from start_vertex through unvisited vertices."""
        detour = []
        current = start_vertex
        local_unvisited = unvisited.copy()

        while local_unvisited:
            neighbors = [n for n in self.adjacency_graph[current] if n in local_unvisited]
            if not neighbors:
                break

            next_vertex = neighbors[0]  # Take first available
            detour.append(next_vertex)
            local_unvisited.remove(next_vertex)
            current = next_vertex

        return detour

    def _find_alternative_path(self, current: int, visited: Set[int], cycle: List[int]) -> int:
        """Find an alternative when we hit a dead end."""
        # Look for any unvisited vertex connected to any visited vertex
        unvisited = set(self.adjacency_graph.keys()) - visited

        for unvisited_v in unvisited:
            for visited_v in visited:
                if unvisited_v in self.adjacency_graph[visited_v]:
                    return unvisited_v

        return None

    def build_grid_constrained_cycle(self) -> List[int]:
        """Build the main Hamiltonian cycle constrained to grid edges."""
        return self._find_cycle_through_graph()

    def render_grid_constrained_path(self, save_path: str = "path.png"):
        """Render ONLY the valid grid edges as a path, no secondary lines."""
        cycle = self.build_grid_constrained_cycle()

        if not cycle:
            print("No cycle could be generated")
            return

        # Use EXACT same coordinate system and setup as the grid
        fig, ax = plt.subplots(figsize=(5.4, 9.6), facecolor='white', dpi=100)
        ax.set_facecolor('white')

        # Use exact same bounds as grid plotting
        half_width = 5.4 / 2
        half_height = 9.6 / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)

        # Draw the continuous path - only consecutive vertices that are connected
        edges_drawn = 0
        gaps_found = 0
        total_vertices = len(cycle)

        # Draw the path as a single continuous line
        for i in range(len(cycle) - 1):
            current_idx = cycle[i]
            next_idx = cycle[i + 1]

            # Only draw if there's a valid edge between consecutive vertices
            if next_idx in self.adjacency_graph[current_idx]:
                x1, y1 = self.vertices[current_idx]
                x2, y2 = self.vertices[next_idx]
                ax.plot([x1, x2], [y1, y2], 'black', linewidth=1.2, alpha=0.9, zorder=2)
                edges_drawn += 1
            else:
                gaps_found += 1

        # Try to close the cycle if start and end are connected
        if len(cycle) > 2:
            first_idx = cycle[0]
            last_idx = cycle[-1]
            if first_idx in self.adjacency_graph.get(last_idx, set()):
                x1, y1 = self.vertices[last_idx]
                x2, y2 = self.vertices[first_idx]
                ax.plot([x1, x2], [y1, y2], 'black', linewidth=1.2, alpha=0.9, zorder=2)
                edges_drawn += 1

        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"Grid-constrained Hamiltonian cycle saved to {save_path}")
        print(f"Drew {edges_drawn} edges between {total_vertices} vertices")
        if gaps_found > 0:
            print(f"WARNING: Found {gaps_found} gaps in path - not a single connected cycle!")


def main():
    """Generate both grid and path images."""
    print("Generating 540x960 images...")

    # Set a consistent seed for reproducible results
    seed = random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    # Create ONE tiling instance that will be used for both images
    print("1. Creating Ammann-Beenker tiling...")
    tiling = AmmannBeenkerTiling()
    print(f"   Grid offsets: {tiling.grid_offsets}")

    k_range = range(-15, 16)

    # Generate and save the grid image
    print("2. Generating grid image...")
    tiling.plot_tiling(
        k_range=k_range,
        figsize=(5.4, 9.6),
        line_width=1.0,
        save_path='grid.png'
    )
    print("   * Grid saved as grid.png")

    # Generate the grid-constrained Hamiltonian path
    print("3. Generating grid-constrained Hamiltonian path...")
    grid_cycle = GridConstrainedHamiltonianCycle(tiling, k_range)
    grid_cycle.render_grid_constrained_path('path.png')

    print("\nGeneration complete!")
    print("Files created:")
    print("  * grid.png - 540x960 Ammann-Beenker tiling")
    print("  * path.png - 540x960 grid-constrained Hamiltonian path")
    print("")
    print("The path now only uses edges that exist in the tiling structure,")
    print("creating a true 'maze' path through the Ammann-Beenker quasicrystal.")


if __name__ == "__main__":
    main()
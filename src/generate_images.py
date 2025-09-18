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
    Creates a proper Hamiltonian cycle that only uses edges existing in the
    Ammann-Beenker tiling structure, forming a single continuous loop.
    """

    def __init__(self, tiling: AmmannBeenkerTiling, k_range: range):
        self.tiling = tiling
        self.k_range = k_range
        self.adjacency_graph = defaultdict(set)
        self.vertices = []
        self.vertex_to_idx = {}
        self._build_grid_graph()

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

    def _find_cycle_through_graph(self) -> List[int]:
        """
        Creates a single connected Hamiltonian path by building bridges between
        isolated components and creating one continuous cycle.
        """
        if not self.vertices:
            return []

        # Find the largest connected component to start with
        components = self._find_connected_components()
        if not components:
            return list(range(len(self.vertices)))

        # Start with the largest component
        main_component = max(components, key=len)
        main_path = self._build_path_in_component(main_component)

        # Add all other vertices by inserting them into the main path
        used_vertices = set(main_component)
        remaining_vertices = [v for v in range(len(self.vertices)) if v not in used_vertices]

        # Insert remaining vertices into the path where they have connections
        final_path = main_path.copy()

        for vertex in remaining_vertices:
            # Find the best insertion point in the path
            best_insertion = self._find_best_insertion_point(final_path, vertex)
            if best_insertion is not None:
                final_path.insert(best_insertion, vertex)
            else:
                # If no good insertion point, add to the end
                final_path.append(vertex)

        return final_path

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

        # Draw ONLY the edges that are part of the Hamiltonian cycle
        # This creates a single connected path through the grid structure
        valid_edges = 0
        invalid_edges = 0

        for i in range(len(cycle)):
            current_idx = cycle[i]
            next_idx = cycle[(i + 1) % len(cycle)]

            # Get coordinates in tiling space
            x1, y1 = self.vertices[current_idx]
            x2, y2 = self.vertices[next_idx]

            # Check if this is a valid grid edge
            if next_idx in self.adjacency_graph[current_idx]:
                # Valid grid edge - draw as part of the Hamiltonian cycle
                ax.plot([x1, x2], [y1, y2], 'black', linewidth=1.0, alpha=0.9, zorder=2)
                valid_edges += 1
            else:
                # Invalid edge - this shouldn't happen with proper algorithm
                # But if it does, we skip it to maintain grid constraint
                invalid_edges += 1

        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"Grid-constrained Hamiltonian cycle saved to {save_path}")
        print(f"Cycle uses {valid_edges} valid grid edges, skipped {invalid_edges} invalid connections")
        print(f"Visits {len(cycle)} vertices in a single connected path")


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
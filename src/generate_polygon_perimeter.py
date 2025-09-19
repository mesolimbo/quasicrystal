#!/usr/bin/env python3
"""Generate perimeter path by identifying outer polygon edges.

Human approach:
1. Look at each polygon (square/rhombus)
2. Find which polygons are on the outside (have edges with no neighbors)
3. Create a path from all the edges that don't have neighbors

This gives us the actual perimeter outline.
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque

# Import our modules
from debruijn_tiling import AmmannBeenkerTiling, generate_random_tiling

WIDTH = 540
HEIGHT = 960


class PolygonPerimeterPath:
    """
    Creates perimeter path by finding outer polygon edges (edges with no neighbors).
    """

    def __init__(self, tiling: AmmannBeenkerTiling, k_range: range):
        self.tiling = tiling
        self.k_range = k_range

        # Polygon data
        self.squares = []
        self.rhombi = []
        self.all_polygons = []

        # Edge tracking
        self.all_edges = set()  # All edges in the tiling
        self.edge_count = defaultdict(int)  # How many polygons share each edge
        self.outer_edges = set()  # Edges that belong to only one polygon

        self._extract_polygons()
        self._find_outer_edges()

    def _extract_polygons(self):
        """Extract all polygons from the tiling."""
        print("Extracting polygons from tiling...")

        self.squares, self.rhombi = self.tiling.generate_tiles(self.k_range, width=5.4, height=9.6)

        # Normalize polygon vertices
        self.squares = [[(round(v[0], 6), round(v[1], 6)) for v in square] for square in self.squares]
        self.rhombi = [[(round(v[0], 6), round(v[1], 6)) for v in rhombus] for rhombus in self.rhombi]

        self.all_polygons = self.squares + self.rhombi

        print(f"Found {len(self.squares)} squares and {len(self.rhombi)} rhombi")
        print(f"Total {len(self.all_polygons)} polygons")

    def _find_outer_edges(self):
        """Find edges that belong to only one polygon (outer edges)."""
        print("Finding outer edges...")

        # Count how many polygons each edge belongs to
        for polygon in self.all_polygons:
            # Create edges from consecutive vertices
            for i in range(len(polygon)):
                v1 = polygon[i]
                v2 = polygon[(i + 1) % len(polygon)]
                edge = tuple(sorted([v1, v2]))  # Normalize edge direction

                self.all_edges.add(edge)
                self.edge_count[edge] += 1

        # Outer edges are those that appear in only one polygon
        self.outer_edges = {edge for edge, count in self.edge_count.items() if count == 1}

        print(f"Total edges: {len(self.all_edges)}")
        print(f"Outer edges: {len(self.outer_edges)}")

    def _build_perimeter_path(self):
        """Build a connected path following the outer edges."""
        if not self.outer_edges:
            return []

        print(f"Building perimeter path from {len(self.outer_edges)} outer edges...")

        # Build adjacency graph from outer edges
        edge_adj = defaultdict(list)
        vertices = set()

        for v1, v2 in self.outer_edges:
            edge_adj[v1].append(v2)
            edge_adj[v2].append(v1)
            vertices.add(v1)
            vertices.add(v2)

        print(f"Outer edge graph has {len(vertices)} vertices")

        # Find connected components in outer edge graph
        components = self._find_connected_components(edge_adj, vertices)
        print(f"Found {len(components)} perimeter components")

        if not components:
            return []

        # Use largest component for perimeter
        largest_component = max(components, key=len)
        print(f"Largest perimeter component has {len(largest_component)} vertices")

        # Build path through this component
        perimeter_path = self._trace_perimeter_path(largest_component, edge_adj)

        print(f"Perimeter path has {len(perimeter_path)} vertices")
        return perimeter_path

    def _find_connected_components(self, adj_graph, all_vertices):
        """Find connected components in the outer edge graph."""
        visited = set()
        components = []

        for vertex in all_vertices:
            if vertex not in visited:
                component = []
                stack = [vertex]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        for neighbor in adj_graph[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if component:
                    components.append(component)

        return components

    def _trace_perimeter_path(self, component_vertices, edge_adj):
        """Trace a path around the perimeter component."""
        if not component_vertices:
            return []

        # Find a corner vertex to start from (vertex with degree 2 or extreme position)
        start_vertex = self._find_start_vertex(component_vertices, edge_adj)

        # Walk around the perimeter
        path = []
        current = start_vertex
        visited_edges = set()

        while current is not None:
            path.append(current)

            # Find next unvisited neighbor
            next_vertex = None
            for neighbor in edge_adj[current]:
                edge = tuple(sorted([current, neighbor]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    next_vertex = neighbor
                    break

            # If this is a closed loop and we're back at start, stop
            if next_vertex == start_vertex and len(path) > 2:
                break

            current = next_vertex

        return path

    def _find_start_vertex(self, component_vertices, edge_adj):
        """Find a good starting vertex for perimeter tracing."""
        if not component_vertices:
            return None

        # Prefer vertices with degree 2 (corners)
        corner_vertices = [v for v in component_vertices if len(edge_adj[v]) == 2]

        if corner_vertices:
            # Choose corner vertex with most extreme position
            return self._find_extreme_vertex(corner_vertices)
        else:
            # No clear corners, just pick extreme vertex
            return self._find_extreme_vertex(component_vertices)

    def _find_extreme_vertex(self, vertices):
        """Find vertex at extreme position (e.g., leftmost, then topmost)."""
        if not vertices:
            return None

        # Find leftmost vertex
        min_x = min(v[0] for v in vertices)
        leftmost_vertices = [v for v in vertices if v[0] == min_x]

        # Among leftmost, find topmost
        max_y = max(v[1] for v in leftmost_vertices)
        extreme_vertex = next(v for v in leftmost_vertices if v[1] == max_y)

        return extreme_vertex

    def render_perimeter_path(self, save_path: str = "path.png"):
        """Render the perimeter path."""
        perimeter_path = self._build_perimeter_path()

        if not perimeter_path:
            print("No perimeter path could be generated")
            return

        # Use same coordinate system as grid
        fig, ax = plt.subplots(figsize=(5.4, 9.6), facecolor='white', dpi=100)
        ax.set_facecolor('white')

        # Use exact same bounds as grid plotting
        half_width = 5.4 / 2
        half_height = 9.6 / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)

        edges_drawn = 0
        total_segments = len(perimeter_path) - 1

        # Draw the perimeter path
        for i in range(len(perimeter_path) - 1):
            current_vertex = perimeter_path[i]
            next_vertex = perimeter_path[i + 1]

            # Check if this is actually an outer edge
            edge = tuple(sorted([current_vertex, next_vertex]))
            if edge in self.outer_edges:
                x1, y1 = current_vertex
                x2, y2 = next_vertex
                ax.plot([x1, x2], [y1, y2], 'black', linewidth=2.0, alpha=0.9, zorder=2)
                edges_drawn += 1

        # Try to close the loop if start and end are connected by an outer edge
        if len(perimeter_path) > 2:
            first_vertex = perimeter_path[0]
            last_vertex = perimeter_path[-1]
            closing_edge = tuple(sorted([first_vertex, last_vertex]))

            if closing_edge in self.outer_edges:
                x1, y1 = last_vertex
                x2, y2 = first_vertex
                ax.plot([x1, x2], [y1, y2], 'black', linewidth=2.0, alpha=0.9, zorder=2)
                edges_drawn += 1

        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"Perimeter path saved to {save_path}")
        print(f"Drew {edges_drawn} outer edges through {len(perimeter_path)} vertices")

        # Calculate coverage
        if edges_drawn > 0:
            coverage = 100 * edges_drawn / len(self.outer_edges)
            print(f"Coverage: {edges_drawn}/{len(self.outer_edges)} outer edges ({coverage:.1f}%)")


def main():
    """Generate both grid and perimeter path images."""
    print("Generating 540x960 polygon perimeter path...")

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

    # Generate the perimeter path
    print("3. Generating polygon perimeter path...")
    perimeter_path = PolygonPerimeterPath(tiling, k_range)
    perimeter_path.render_perimeter_path('path.png')

    print("\nGeneration complete!")
    print("Files created:")
    print("  * grid.png - 540x960 Ammann-Beenker tiling")
    print("  * path.png - 540x960 polygon perimeter path")
    print("")
    print("The path traces the outer boundary using edges that don't have neighbors.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Generate concentric polygon layers using iterative peeling.

Algorithm:
1. Start with outer layer: polygons with edges that have no neighbors
2. "Hairy" edges: any edge containing a vertex from outer layer polygons
3. "Hairy" polygons: polygons that have any hairy edges
4. Next layer: polygons that touch hairy polygons but aren't hairy themselves
5. Repeat until no more layers can be found

This creates clean concentric loops by peeling polygon layers.
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


class PolygonLayerPath:
    """
    Creates concentric polygon layers using iterative peeling algorithm.
    """

    def __init__(self, tiling: AmmannBeenkerTiling, k_range: range):
        self.tiling = tiling
        self.k_range = k_range

        # Polygon data
        self.squares = []
        self.rhombi = []
        self.all_polygons = []
        self.polygon_edges = {}  # polygon_idx -> list of edges

        # Edge tracking
        self.all_edges = set()  # All edges in the tiling
        self.edge_count = defaultdict(int)  # How many polygons share each edge
        self.edge_to_polygons = defaultdict(list)  # edge -> list of polygon indices

        # Layer tracking
        self.polygon_layers = {}  # polygon_idx -> layer_number
        self.layers = []  # List of sets, each containing polygon indices in that layer

        self._extract_polygons()
        self._build_edge_mappings()
        self._compute_polygon_layers()

    def _extract_polygons(self):
        """Extract all polygons from the tiling."""
        print("Extracting polygons from tiling...")

        self.squares, self.rhombi = self.tiling.generate_tiles(self.k_range, width=5.4, height=9.6)

        # Normalize polygon vertices
        self.squares = [[(round(v[0], 6), round(v[1], 6)) for v in square] for square in self.squares]
        self.rhombi = [[(round(v[0], 6), round(v[1], 6)) for v in rhombus] for rhombus in self.rhombi]

        self.all_polygons = self.squares + self.rhombi

        # Create polygon edge mappings
        for poly_idx, polygon in enumerate(self.all_polygons):
            edges = []
            for i in range(len(polygon)):
                v1 = polygon[i]
                v2 = polygon[(i + 1) % len(polygon)]
                edge = tuple(sorted([v1, v2]))
                edges.append(edge)
            self.polygon_edges[poly_idx] = edges

        print(f"Found {len(self.squares)} squares and {len(self.rhombi)} rhombi")
        print(f"Total {len(self.all_polygons)} polygons")

    def _build_edge_mappings(self):
        """Build mappings between edges and polygons."""
        print("Building edge mappings...")

        for poly_idx, edges in self.polygon_edges.items():
            for edge in edges:
                self.all_edges.add(edge)
                self.edge_count[edge] += 1
                self.edge_to_polygons[edge].append(poly_idx)

        print(f"Total edges: {len(self.all_edges)}")

    def _compute_polygon_layers(self):
        """Compute polygon layers using flood-fill coloring by adjacency."""
        print("Computing polygon layers using flood-fill coloring...")

        # Step 1: Find outer perimeter polygons
        outer_perimeter = self._find_perimeter_in_available(set(range(len(self.all_polygons))))
        print(f"Found {len(outer_perimeter)} outer perimeter polygons")

        # Step 2: Flood-fill by polygon adjacency distance from perimeter
        uncolored_polygons = set(range(len(self.all_polygons))) - outer_perimeter
        current_layer = outer_perimeter
        layer_num = 0

        while current_layer:
            # Assign current layer
            self.layers.append(current_layer.copy())
            for poly_idx in current_layer:
                self.polygon_layers[poly_idx] = layer_num

            print(f"Layer {layer_num}: {len(current_layer)} polygons")

            if not uncolored_polygons:
                break

            # Find next layer: all uncolored polygons that touch current layer
            next_layer = set()
            for poly_idx in uncolored_polygons:
                if self._polygon_touches_any_in_set(poly_idx, current_layer):
                    next_layer.add(poly_idx)

            # Remove next layer from uncolored
            uncolored_polygons -= next_layer
            current_layer = next_layer
            layer_num += 1

            # Safety limit
            if layer_num > 50:
                break

        print(f"\nCreated {len(self.layers)} polygon layers using flood-fill")

    def _find_outer_layer_polygons(self, remaining_polygons):
        """Find polygons with edges that have no neighbors (outer boundary)."""
        outer_polygons = set()

        for poly_idx in remaining_polygons:
            edges = self.polygon_edges[poly_idx]

            # Check if any edge has no neighbors (count == 1)
            has_outer_edge = any(self.edge_count[edge] == 1 for edge in edges)

            if has_outer_edge:
                outer_polygons.add(poly_idx)

        return outer_polygons

    def _find_perimeter_in_available(self, available_polygons):
        """Find polygons that form the true outer perimeter (edges with no neighbors globally)."""
        if not available_polygons:
            return set()

        # Find polygons with edges that have no neighbors in the GLOBAL tiling
        # (not just within available polygons)
        perimeter_polygons = set()

        for poly_idx in available_polygons:
            edges = self.polygon_edges[poly_idx]

            # Check if any edge has no neighbors globally (count == 1 in original tiling)
            has_boundary_edge = any(self.edge_count[edge] == 1 for edge in edges)

            if has_boundary_edge:
                perimeter_polygons.add(poly_idx)

        return perimeter_polygons

    def _grow_hairs_from_perimeter(self, perimeter_polygons, available_polygons):
        """Grow hairs from perimeter vertices to create insulation layer."""
        if not perimeter_polygons:
            return set()

        # Get all vertices from perimeter polygons
        perimeter_vertices = set()
        for poly_idx in perimeter_polygons:
            polygon = self.all_polygons[poly_idx]
            perimeter_vertices.update(polygon)

        # Find "hairy" edges: edges that contain any perimeter vertex
        hairy_edges = set()
        for edge in self.all_edges:
            v1, v2 = edge
            if v1 in perimeter_vertices or v2 in perimeter_vertices:
                hairy_edges.add(edge)

        # Find "hairy" polygons: available polygons that have any hairy edges
        # (excluding the perimeter polygons themselves)
        hairy_polygons = set()
        for poly_idx in available_polygons:
            if poly_idx in perimeter_polygons:
                continue  # Skip perimeter polygons

            edges = self.polygon_edges[poly_idx]
            has_hairy_edge = any(edge in hairy_edges for edge in edges)

            if has_hairy_edge:
                hairy_polygons.add(poly_idx)

        return hairy_polygons

    def _find_polygons_sharing_vertices_with_perimeter(self, perimeter_polygons, available_polygons):
        """Find all available polygons that share any vertex with perimeter polygons."""
        if not perimeter_polygons:
            return set()

        # Get all vertices from perimeter polygons
        perimeter_vertices = set()
        for poly_idx in perimeter_polygons:
            polygon = self.all_polygons[poly_idx]
            perimeter_vertices.update(polygon)

        # Find available polygons that share any vertex with perimeter
        sharing_polygons = set()
        for poly_idx in available_polygons:
            if poly_idx in perimeter_polygons:
                continue  # Skip perimeter polygons themselves

            polygon = self.all_polygons[poly_idx]
            polygon_vertices = set(polygon)

            # Check if this polygon shares any vertex with perimeter
            if polygon_vertices & perimeter_vertices:  # Intersection
                sharing_polygons.add(poly_idx)

        return sharing_polygons

    def _polygon_touches_any_in_set(self, poly_idx, polygon_set):
        """Check if polygon touches any polygon in the given set."""
        if not polygon_set:
            return False

        poly_edges = self.polygon_edges[poly_idx]

        for edge in poly_edges:
            # Get all polygons sharing this edge
            sharing_polygons = self.edge_to_polygons[edge]

            # Check if any sharing polygon is in the target set
            if any(shared_poly in polygon_set for shared_poly in sharing_polygons if shared_poly != poly_idx):
                return True

        return False

    def _build_layer_perimeter_path(self, layer_polygons):
        """Build path through all edges within this layer of polygons."""
        if not layer_polygons:
            return []

        # Instead of trying to find "boundary" of layer, build path through all edges in layer
        layer_edges = set()
        for poly_idx in layer_polygons:
            layer_edges.update(self.polygon_edges[poly_idx])

        # Build adjacency graph from all edges in this layer
        edge_adj = defaultdict(list)
        vertices = set()

        for v1, v2 in layer_edges:
            edge_adj[v1].append(v2)
            edge_adj[v2].append(v1)
            vertices.add(v1)
            vertices.add(v2)

        # Find connected components
        components = self._find_connected_components(edge_adj, vertices)

        if not components:
            return []

        # Use largest component and build a path through it
        largest_component = max(components, key=len)

        # Build aggressive path through the component
        return self._build_aggressive_path_in_component(largest_component, edge_adj)

    def _build_aggressive_path_in_component(self, vertices, edge_adj):
        """Build longest possible path through all vertices in component."""
        if not vertices:
            return []

        # Start from vertex with lowest degree
        start_vertex = min(vertices, key=lambda v: len(edge_adj[v]))

        visited = set()
        path = []

        def dfs(vertex):
            visited.add(vertex)
            path.append(vertex)

            # Visit unvisited neighbors, preferring those with fewer unvisited neighbors
            neighbors = [n for n in edge_adj[vertex] if n not in visited]
            neighbors.sort(key=lambda v: len([nn for nn in edge_adj[v] if nn not in visited]))

            for neighbor in neighbors:
                dfs(neighbor)

        dfs(start_vertex)
        return path

    def _build_layer_path_original_edges_only(self, layer_polygons):
        """Build path using only original grid edges within this layer."""
        if not layer_polygons:
            return []

        # Get all vertices from polygons in this layer
        layer_vertices = set()
        for poly_idx in layer_polygons:
            polygon = self.all_polygons[poly_idx]
            layer_vertices.update(polygon)

        # Build adjacency using only original grid edges between layer vertices
        layer_adj = defaultdict(list)
        for edge in self.all_edges:
            v1, v2 = edge
            if v1 in layer_vertices and v2 in layer_vertices:
                layer_adj[v1].append(v2)
                layer_adj[v2].append(v1)

        # Find connected components
        components = self._find_connected_components(layer_adj, layer_vertices)

        if not components:
            return []

        # Use largest component
        largest_component = max(components, key=len)

        # Build path through this component using original edges only
        return self._build_aggressive_path_in_component(largest_component, layer_adj)

    def _find_connected_components(self, adj_graph, all_vertices):
        """Find connected components in the graph."""
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

        # Find extreme vertex to start from
        start_vertex = min(component_vertices, key=lambda v: (v[0], v[1]))

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

    def render_polygon_layers(self, save_path: str = "path.png"):
        """Render polygons filled with solid colors based on their layer."""
        if not self.layers:
            print("No polygon layers found")
            return

        # Use same coordinate system as grid
        fig, ax = plt.subplots(figsize=(5.4, 9.6), facecolor='white', dpi=100)
        ax.set_facecolor('white')

        # Use exact same bounds as grid plotting
        half_width = 5.4 / 2
        half_height = 9.6 / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)

        # Calculate grey values: 100% - (layer_number * 100/15)%
        colors = []
        for i in range(len(self.layers)):
            # Layer i gets grey value = 100 - (i+1)*100/15 percent
            grey_percentage = 100 - ((i + 1) * 100 / 15)
            grey_value = max(0, grey_percentage / 100)  # Convert to 0-1 range, don't go below 0
            colors.append(str(grey_value))

        total_polygons = 0

        # Fill polygons with solid colors based on their layer
        for layer_idx, layer_polygons in enumerate(self.layers):
            if not layer_polygons:
                continue

            color = colors[layer_idx % len(colors)]
            alpha = 0.8

            print(f"Filling layer {layer_idx} with {len(layer_polygons)} polygons in color {color}")

            # Fill each polygon in this layer
            for poly_idx in layer_polygons:
                polygon = self.all_polygons[poly_idx]

                # Create matplotlib polygon patch
                xs = [vertex[0] for vertex in polygon]
                ys = [vertex[1] for vertex in polygon]

                # Fill the polygon
                ax.fill(xs, ys, color=color, alpha=alpha, edgecolor='white', linewidth=0.5, zorder=1)

            total_polygons += len(layer_polygons)

        # No path edges drawn - just solid colored polygons

        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"\nPolygon layers saved to {save_path}")
        print(f"Total: {total_polygons} polygons filled with greyscale gradient")


def main():
    """Generate both grid and polygon layer paths."""
    print("Generating 540x960 polygon layer paths...")

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

    # Generate the polygon layer paths
    print("3. Generating polygon layer paths...")
    layer_paths = PolygonLayerPath(tiling, k_range)
    layer_paths.render_polygon_layers('path.png')

    print("\nGeneration complete!")
    print("Files created:")
    print("  * grid.png - 540x960 Ammann-Beenker tiling")
    print("  * path.png - 540x960 polygon layer paths")
    print("")
    print("The paths show concentric loops created by iterative polygon peeling.")


if __name__ == "__main__":
    main()
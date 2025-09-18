import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import List, Tuple, Optional, Set
import math

class AmmannBeenkerTiling:
    """
    Generates Ammann-Beenker tilings with 8-fold rotational symmetry.
    Creates harmonious black and white line renderings using only squares and rhombi.
    """

    def __init__(self):
        """Initialize the Ammann-Beenker tiling generator."""
        self.n_grids = 4  # For 8-fold symmetry, use 4 grids (not 8 to avoid duplicates)
        self.basis_vectors = self._generate_basis_vectors()
        self.grid_offsets = self._generate_random_offsets()

    def _generate_basis_vectors(self) -> np.ndarray:
        """Generate basis vectors for 8-fold symmetry."""
        vectors = []
        # For 8-fold symmetry, use 4 directions spaced by π/4
        # Keep full scale - we'll make tiles smaller by using denser grid spacing
        for j in range(self.n_grids):
            angle = j * np.pi / 4  # π/4 spacing for 8-fold
            vector = np.array([np.cos(angle), np.sin(angle)])
            vectors.append(vector)
        return np.array(vectors)

    def _generate_random_offsets(self) -> np.ndarray:
        """Generate random grid offsets for variation."""
        # Use smaller random offsets for better tile formation
        offsets = np.random.uniform(0.2, 0.8, self.n_grids)
        return offsets

    def _find_intersections(self, k_range: range) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
        """Find all intersections between pairs of grids."""
        intersections = []

        # Compare each pair of grids
        for i in range(self.n_grids):
            for j in range(i + 1, self.n_grids):
                e_i, e_j = self.basis_vectors[i], self.basis_vectors[j]
                gamma_i, gamma_j = self.grid_offsets[i], self.grid_offsets[j]

                # Create coefficient matrix for line intersection
                coeff_matrix = np.array([[e_i[0], e_i[1]],
                                       [e_j[0], e_j[1]]])

                # Skip if lines are parallel
                det = np.linalg.det(coeff_matrix)
                if abs(det) < 1e-10:
                    continue

                inv_coeff = np.linalg.inv(coeff_matrix)

                # Find intersections for all line index combinations
                # Scale the grid spacing to make tiles smaller (4x denser grid)
                grid_scale = 0.25
                for k_i in k_range:
                    for k_j in k_range:
                        rhs = np.array([(k_i + gamma_i) * grid_scale, (k_j + gamma_j) * grid_scale])
                        intersection = inv_coeff @ rhs
                        intersections.append((intersection, (i, j), (k_i, k_j)))

        return intersections

    def _classify_tile(self, vertices: np.ndarray) -> str:
        """Classify whether a tile is a square or rhombus based on its angles."""
        if len(vertices) != 4:
            return "invalid"

        # Calculate side lengths
        sides = []
        for i in range(4):
            side = np.linalg.norm(vertices[(i+1) % 4] - vertices[i])
            sides.append(side)

        # Calculate angles
        angles = []
        for i in range(4):
            v1 = vertices[(i-1) % 4] - vertices[i]
            v2 = vertices[(i+1) % 4] - vertices[i]
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            # Calculate angle
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

        # Check if it's approximately a square (all angles ≈ 90°)
        right_angle = np.pi / 2
        angle_tolerance = 0.3  # radians

        square_angles = sum(1 for angle in angles if abs(angle - right_angle) < angle_tolerance)

        if square_angles >= 3:  # If at least 3 angles are close to 90°
            return "square"
        else:
            return "rhombus"

    def _get_tile_vertices(self, intersection: np.ndarray, grid_pair: Tuple[int, int],
                          k_indices: Tuple[int, int]) -> np.ndarray:
        """Get the four vertices of the tile around an intersection."""
        i, j = grid_pair
        k_i, k_j = k_indices

        # Calculate the full index vector for this intersection
        index_vector = np.zeros(self.n_grids)

        # Set known indices for intersecting grids
        index_vector[i] = k_i
        index_vector[j] = k_j

        # Calculate indices for other grids using ceiling function
        # Use the same grid scale as in intersection finding
        grid_scale = 0.25
        for grid_idx in range(self.n_grids):
            if grid_idx != i and grid_idx != j:
                dot_product = np.dot(self.basis_vectors[grid_idx], intersection)
                scaled_index = (dot_product - self.grid_offsets[grid_idx] * grid_scale) / grid_scale
                index_vector[grid_idx] = np.ceil(scaled_index)

        # Generate the four vertices of the tile
        vertices = []
        for di in [0, 1]:
            for dj in [0, 1]:
                vertex_indices = index_vector.copy()
                vertex_indices[i] += di
                vertex_indices[j] += dj

                # Convert indices to position with grid scaling
                position = np.sum([k * self.basis_vectors[idx] * grid_scale
                                 for idx, k in enumerate(vertex_indices)], axis=0)
                vertices.append(position)

        return np.array(vertices)

    def _order_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Order vertices clockwise around their centroid."""
        center = np.mean(vertices, axis=0)

        # Calculate angles from center to each vertex
        angles = []
        for vertex in vertices:
            diff = vertex - center
            angle = np.arctan2(diff[1], diff[0])
            angles.append(angle)

        # Sort vertices by angle
        sorted_indices = np.argsort(angles)
        return vertices[sorted_indices]

    def generate_tiles(self, k_range: Optional[range] = None,
                      width: float = 5.4, height: float = 9.6,
                      center: Tuple[float, float] = (0, 0),
                      margin_ratio: float = 0.01) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate tiles using the Ammann-Beenker method to fill a rectangle with margins.

        Args:
            k_range: Range of grid line indices
            width: Width of the rectangle to fill
            height: Height of the rectangle to fill
            center: Center point of the rectangle
            margin_ratio: Fraction of width/height to use as margin (0.02 = 2% margin)

        Returns:
            Tuple of (squares, rhombi) - each as list of 4 vertices
        """
        if k_range is None:
            k_range = range(-15, 16)  # Back to reasonable range with grid scaling

        intersections = self._find_intersections(k_range)
        squares = []
        rhombi = []
        center_point = np.array(center)

        # Calculate small margins to avoid cut-off polygons at edges
        x_margin = width * margin_ratio
        y_margin = height * margin_ratio

        # Define the main rectangle bounds (most of the area)
        main_x_min = center_point[0] - width/2 + x_margin
        main_x_max = center_point[0] + width/2 - x_margin
        main_y_min = center_point[1] - height/2 + y_margin
        main_y_max = center_point[1] + height/2 - y_margin

        # Extended bounds for tile generation
        extended_margin = 1.0
        x_min = center_point[0] - width/2 - extended_margin
        x_max = center_point[0] + width/2 + extended_margin
        y_min = center_point[1] - height/2 - extended_margin
        y_max = center_point[1] + height/2 + extended_margin

        for intersection, grid_pair, k_indices in intersections:
            # Skip intersections way outside our extended area
            if (intersection[0] < x_min - 1 or intersection[0] > x_max + 1 or
                intersection[1] < y_min - 1 or intersection[1] > y_max + 1):
                continue

            vertices = self._get_tile_vertices(intersection, grid_pair, k_indices)

            # Check if tile overlaps with our main rectangle area
            tile_x_min = np.min(vertices[:, 0])
            tile_x_max = np.max(vertices[:, 0])
            tile_y_min = np.min(vertices[:, 1])
            tile_y_max = np.max(vertices[:, 1])

            # Only include tiles that are completely within the rectangle bounds
            # (no polygons poking out beyond 540x960)
            rect_x_min = center_point[0] - width/2
            rect_x_max = center_point[0] + width/2
            rect_y_min = center_point[1] - height/2
            rect_y_max = center_point[1] + height/2

            if (tile_x_min >= rect_x_min and tile_x_max <= rect_x_max and
                tile_y_min >= rect_y_min and tile_y_max <= rect_y_max):

                # Order vertices properly
                vertices = self._order_vertices(vertices)

                # Classify the tile
                tile_type = self._classify_tile(vertices)

                if tile_type == "square":
                    squares.append(vertices)
                elif tile_type == "rhombus":
                    rhombi.append(vertices)

        return squares, rhombi

    def plot_tiling(self, k_range: Optional[range] = None,
                   figsize: Tuple[float, float] = (5.4, 9.6),
                   line_width: float = 1.0,
                   save_path: Optional[str] = None) -> None:
        """
        Generate and plot a harmonious black and white Ammann-Beenker tiling.
        Fills the entire 540x960 rectangle and clips to exact dimensions.
        """
        if k_range is None:
            k_range = range(-15, 16)

        # Generate tiles to fill the rectangle
        squares, rhombi = self.generate_tiles(k_range, width=figsize[0], height=figsize[1])

        # Create figure with white background - exact pixel dimensions
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', dpi=100)
        ax.set_facecolor('white')

        # Set exact rectangle bounds to match 540x960 output
        half_width = figsize[0] / 2
        half_height = figsize[1] / 2
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)

        # Plot squares (unfilled, black outline)
        for vertices in squares:
            square = patches.Polygon(vertices, fill=False, edgecolor='black',
                                   linewidth=line_width, alpha=1.0)
            ax.add_patch(square)

        # Plot rhombi (unfilled, black outline)
        for vertices in rhombi:
            rhombus = patches.Polygon(vertices, fill=False, edgecolor='black',
                                    linewidth=line_width, alpha=1.0)
            ax.add_patch(rhombus)

        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.axis('off')

        # Remove all margins and padding to ensure exact 540x960 pixels
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save if path provided with exact dimensions
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                       facecolor='white', edgecolor='none')

        # Only show if no save path provided
        if not save_path:
            plt.show()
        else:
            plt.close()

        print(f"Generated {len(squares)} squares and {len(rhombi)} rhombi")
        print(f"Rectangle dimensions: {figsize[0]} x {figsize[1]} (should be 5.4 x 9.6)")
        print(f"Output pixels: 540 x 960")


def generate_random_tiling(size: str = 'medium', save_path: Optional[str] = None) -> None:
    """
    Generate a random Ammann-Beenker tiling with different size options.
    """
    # Configure parameters based on size (all maintain 540x960 aspect ratio)
    size_configs = {
        'small': {'k_range': range(-12, 13), 'figsize': (5.4, 9.6)},
        'medium': {'k_range': range(-15, 16), 'figsize': (5.4, 9.6)},
        'large': {'k_range': range(-18, 19), 'figsize': (5.4, 9.6)}
    }

    config = size_configs.get(size, size_configs['medium'])

    # Create tiling generator with random seed
    random.seed()
    np.random.seed()

    tiling = AmmannBeenkerTiling()

    print(f"Generating {size} random Ammann-Beenker tiling...")
    print(f"Grid offsets: {tiling.grid_offsets}")

    tiling.plot_tiling(
        k_range=config['k_range'],
        figsize=config['figsize'],
        line_width=1.0,
        save_path=save_path
    )


if __name__ == "__main__":
    # Generate a random tiling
    generate_random_tiling('medium', 'random_ammann_beenker_tiling.png')

    # Generate another different one
    print("\nGenerating another random variation...")
    generate_random_tiling('medium')
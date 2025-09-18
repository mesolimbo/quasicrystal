"""Generate a Hamiltonian path through an Ammann-Beenker tiling for 540×960 display.

This script generates an Ammann-Beenker tiling using the de Bruijn dual grid method,
extracts vertex coordinates from the tiling, and creates a Hamiltonian-style path
that visits each vertex exactly once. The result is rendered as a maze-like path
through the quasicrystalline structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple, Sequence
from pathlib import Path
import argparse

# Import our tiling generator
from debruijn_tiling import AmmannBeenkerTiling

WIDTH = 540
HEIGHT = 960
MARGIN = 12  # pixel padding around the drawing region


def extract_vertices_from_tiling(tiling: AmmannBeenkerTiling,
                               k_range: range = None) -> List[Tuple[float, float]]:
    """Extract all unique vertices from the Ammann-Beenker tiling."""
    if k_range is None:
        k_range = range(-15, 16)

    # Generate tiles
    squares, rhombi = tiling.generate_tiles(k_range, width=5.4, height=9.6)

    # Collect all unique vertices
    vertex_set = set()

    # Add vertices from squares
    for square in squares:
        for vertex in square:
            # Round to avoid floating point precision issues
            vertex_tuple = (round(vertex[0], 6), round(vertex[1], 6))
            vertex_set.add(vertex_tuple)

    # Add vertices from rhombi
    for rhombus in rhombi:
        for vertex in rhombus:
            # Round to avoid floating point precision issues
            vertex_tuple = (round(vertex[0], 6), round(vertex[1], 6))
            vertex_set.add(vertex_tuple)

    # Convert to list and scale to pixel coordinates
    vertices = list(vertex_set)

    # Transform coordinates to fit within 540x960 with margins
    if not vertices:
        raise ValueError("No vertices generated from tiling")

    xs = [x for x, y in vertices]
    ys = [y for x, y in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_x == 0 or span_y == 0:
        raise ValueError("Degenerate vertex dimensions detected")

    # Scale to fit within margins
    scale_x = (WIDTH - 2 * MARGIN) / span_x
    scale_y = (HEIGHT - 2 * MARGIN) / span_y
    offset_x = MARGIN - scale_x * min_x
    offset_y = MARGIN - scale_y * min_y

    scaled_vertices = [(scale_x * x + offset_x, scale_y * y + offset_y)
                      for x, y in vertices]

    return scaled_vertices


def nearest_neighbour_cycle(points: Sequence[Tuple[float, float]],
                          rng: random.Random) -> List[int]:
    """Generate an initial tour using a nearest-neighbour walk."""
    if len(points) < 2:
        return list(range(len(points)))

    unused = list(range(len(points)))
    current = unused.pop(rng.randrange(len(unused)))
    cycle = [current]

    while unused:
        cx, cy = points[current]
        next_index = min(
            unused,
            key=lambda idx: (points[idx][0] - cx) ** 2 + (points[idx][1] - cy) ** 2,
        )
        unused.remove(next_index)
        cycle.append(next_index)
        current = next_index

    return cycle


def two_opt_improvement(cycle: List[int],
                       points: Sequence[Tuple[float, float]],
                       rng: random.Random,
                       max_rounds: int) -> None:
    """Perform a bounded number of 2-opt refinement rounds on the tour."""
    if len(cycle) < 4:
        return

    n = len(cycle)
    for _ in range(max_rounds):
        improved = False
        indices = list(range(n - 1))
        rng.shuffle(indices)
        for i in indices:
            a = cycle[i]
            b = cycle[i + 1]
            ax, ay = points[a]
            bx, by = points[b]
            upper = n if i > 0 else n - 1
            for j in range(i + 2, upper):
                c = cycle[j]
                d = cycle[(j + 1) % n]
                if a == d or b == c:
                    continue
                cx, cy = points[c]
                dx, dy = points[d]
                current_length = math.hypot(ax - bx, ay - by) + math.hypot(cx - dx, cy - dy)
                swapped_length = math.hypot(ax - cx, ay - cy) + math.hypot(bx - dx, by - dy)
                if swapped_length + 1e-9 < current_length:
                    cycle[i + 1 : j + 1] = reversed(cycle[i + 1 : j + 1])
                    improved = True
        if not improved:
            break


def build_cycle(points: Sequence[Tuple[float, float]],
               rng: random.Random,
               two_opt_rounds: int) -> List[int]:
    """Construct a Hamiltonian-style cycle visiting each vertex exactly once."""
    cycle = nearest_neighbour_cycle(points, rng)
    two_opt_improvement(cycle, points, rng, two_opt_rounds)
    return cycle


def render_hamiltonian_path(points: Sequence[Tuple[float, float]],
                           cycle: Sequence[int],
                           show_vertices: bool = True,
                           save_path: str = None) -> None:
    """Render the Hamiltonian path through the tiling vertices."""
    # Convert to inches for exact 540x960 pixels at 100 DPI
    fig_width = WIDTH / 100
    fig_height = HEIGHT / 100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white', dpi=100)
    ax.set_facecolor('white')

    # Set exact pixel bounds
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Flip Y axis to match image coordinates
    ax.invert_yaxis()

    # Draw vertices as small dots if requested
    if show_vertices:
        xs = [x for x, y in points]
        ys = [y for x, y in points]
        ax.scatter(xs, ys, c='lightgray', s=2, alpha=0.6, zorder=1)

    # Draw the Hamiltonian path
    path_x = []
    path_y = []

    # Create closed loop by appending first point at end
    extended_cycle = list(cycle) + [cycle[0]]

    for i in extended_cycle:
        path_x.append(points[i][0])
        path_y.append(points[i][1])

    ax.plot(path_x, path_y, 'black', linewidth=1.2, alpha=0.9, zorder=2)

    # Remove axes and margins
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0,
                   facecolor='white', edgecolor='none')
        print(f"Hamiltonian path saved to {save_path}")

    # Only show if no save path provided
    if not save_path:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k-range-size",
        type=int,
        default=15,
        help="Half-size of k range for tiling generation (default: 15, giving range -15 to 15)"
    )
    parser.add_argument(
        "--two-opt-rounds",
        type=int,
        default=50,
        help="Maximum number of 2-opt refinement sweeps for path optimization"
    )
    parser.add_argument(
        "--show-vertices",
        action="store_true",
        help="Show vertex dots in addition to the path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hamiltonian_path.png",
        help="Output image filename"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    """Main function to generate and render Hamiltonian path through Ammann-Beenker tiling."""
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("Generating Ammann-Beenker tiling...")

    # Create tiling generator
    tiling = AmmannBeenkerTiling()
    k_range = range(-args.k_range_size, args.k_range_size + 1)

    print(f"Using k_range: {k_range}")
    print(f"Grid offsets: {tiling.grid_offsets}")

    # Extract vertices from the tiling
    print("Extracting vertices from tiling...")
    vertices = extract_vertices_from_tiling(tiling, k_range)
    print(f"Extracted {len(vertices)} unique vertices")

    # Build Hamiltonian cycle
    print(f"Building Hamiltonian cycle with {args.two_opt_rounds} 2-opt rounds...")
    rng = random.Random(args.seed)
    cycle = build_cycle(vertices, rng, args.two_opt_rounds)
    print(f"Cycle constructed visiting {len(cycle)} vertices")

    # Calculate total path length
    total_length = 0
    extended_cycle = cycle + [cycle[0]]
    for i in range(len(extended_cycle) - 1):
        p1 = vertices[extended_cycle[i]]
        p2 = vertices[extended_cycle[i + 1]]
        total_length += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    print(f"Total path length: {total_length:.2f} pixels")

    # Render the result
    print(f"Rendering Hamiltonian path...")
    render_hamiltonian_path(
        vertices,
        cycle,
        show_vertices=args.show_vertices,
        save_path=args.output
    )

    print("Done!")


if __name__ == "__main__":
    main()